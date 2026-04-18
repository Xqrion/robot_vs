#!/usr/bin/env python3
"""Hierarchical MAS manager.

Responsibilities:
1) keep slow strategic loop (LeaderAgent),
2) keep fast tactical loop (CarAgent per robot),
3) provide non-blocking state ingestion and latest-task query APIs.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
	from agents import CarAgent, LeaderAgent, plan_cars_concurrently
	from config_loader import ConfigError, ConfigLoader
	from llm_api import AsyncLLMClient
	from memory import LongTermMemory, ShortTermMemory
except ImportError:  # pragma: no cover
	from .agents import CarAgent, LeaderAgent, plan_cars_concurrently  # type: ignore
	from .config_loader import ConfigError, ConfigLoader  # type: ignore
	from .llm_api import AsyncLLMClient  # type: ignore
	from .memory import LongTermMemory, ShortTermMemory  # type: ignore


LOGGER = logging.getLogger("mas.manager")


class SideMASRuntime:
	"""Runtime for one side (red/blue): one Leader + many Cars."""

	def __init__(
		self,
		side: str,
		llm_client: AsyncLLMClient,
		models_cfg: Mapping[str, Any],
		prompts_cfg: Mapping[str, Any],
		ltm_storage_path: Optional[Path] = None,
		stm_window_size: int = 16,
	) -> None:
		normalized_side = _normalize_side(side)
		if not normalized_side:
			raise ValueError("side must be red or blue")

		self.side = normalized_side
		self.llm_client = llm_client
		self.models_cfg = dict(models_cfg)
		self.prompts_cfg = dict(prompts_cfg)

		runtime_cfg = self.models_cfg.get("runtime", {})
		if not isinstance(runtime_cfg, Mapping):
			runtime_cfg = {}
		self.leader_interval_s = max(0.5, _as_float(runtime_cfg.get("leader_loop_interval_s", 5.0), 5.0))
		self.car_interval_s = max(0.2, _as_float(runtime_cfg.get("car_loop_interval_s", 1.0), 1.0))

		self.stm = ShortTermMemory(max_items=max(4, int(stm_window_size)))
		self.ltm = LongTermMemory(storage_path=ltm_storage_path)
		self.leader_agent = LeaderAgent(
			llm_client=self.llm_client,
			models_cfg=self.models_cfg,
			prompts_cfg=self.prompts_cfg,
			stm=self.stm,
			ltm=self.ltm,
			min_cycle_s=self.leader_interval_s,
		)

		self._car_agents: Dict[str, CarAgent] = {}

		self._state_lock = asyncio.Lock()
		self._agent_lock = asyncio.Lock()
		self._decision_lock = asyncio.Lock()
		self._slow_cycle_lock = asyncio.Lock()
		self._fast_cycle_lock = asyncio.Lock()

		self._latest_payload: Dict[str, Any] = {}
		self._latest_battle_state: Dict[str, Any] = {}
		self._robot_ids: List[str] = []

		self._leader_order_text = "Hold formation and preserve survivability."
		self._leader_order_ts_s = 0.0

		self._latest_tasks: Dict[str, Dict[str, Any]] = {}
		self._latest_tasks_ts_s = 0.0

		self._started = False
		self._stop_event = asyncio.Event()
		self._loop_tasks: List[asyncio.Task] = []

	async def start(self) -> None:
		if self._started:
			return

		self._stop_event.clear()
		self._loop_tasks = [
			asyncio.create_task(self._leader_loop(), name="leader_loop_{}".format(self.side)),
			asyncio.create_task(self._car_loop(), name="car_loop_{}".format(self.side)),
		]
		self._started = True
		LOGGER.info(
			"SideMASRuntime started: side=%s leader_interval=%.2fs car_interval=%.2fs",
			self.side,
			self.leader_interval_s,
			self.car_interval_s,
		)

	async def stop(self) -> None:
		if not self._started:
			return

		self._stop_event.set()
		for task in self._loop_tasks:
			task.cancel()
		await asyncio.gather(*self._loop_tasks, return_exceptions=True)
		self._loop_tasks = []
		self._started = False
		LOGGER.info("SideMASRuntime stopped: side=%s", self.side)

	async def ingest_payload(self, payload: Optional[Mapping[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
		safe_payload = _safe_mapping(payload)
		battle_state = _extract_battle_state(safe_payload)
		robot_ids = _extract_robot_ids(safe_payload, battle_state)

		async with self._state_lock:
			self._latest_payload = copy.deepcopy(safe_payload)
			self._latest_battle_state = copy.deepcopy(battle_state)
			if robot_ids:
				self._robot_ids = list(robot_ids)
			elif not self._robot_ids:
				self._robot_ids = _extract_robot_ids({}, battle_state)

			ids_snapshot = list(self._robot_ids)
			state_snapshot = copy.deepcopy(self._latest_battle_state)

		return state_snapshot, ids_snapshot

	async def handle_plan_request(self, payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
		await self.ingest_payload(payload)

		tasks, tasks_ts = await self._get_tasks_snapshot()
		now_s = time.time()
		stale_threshold_s = max(0.8, self.car_interval_s * 1.6)

		_, robot_ids = await self._snapshot_state()
		missing_any = bool(robot_ids and any(robot_id not in tasks for robot_id in robot_ids))
		stale = (not tasks) or (now_s - tasks_ts > stale_threshold_s) or missing_any

		if stale:
			tasks = await self._run_car_cycle_once(force=True)

		tasks = _fill_missing_tasks(tasks, robot_ids)
		leader_order, leader_ts_s = await self._get_leader_snapshot()

		return {
			"tasks": tasks,
			"side": self.side,
			"leader_order": leader_order,
			"meta": {
				"leader_age_s": max(0.0, now_s - leader_ts_s) if leader_ts_s > 0 else None,
				"task_age_s": max(0.0, now_s - tasks_ts) if tasks_ts > 0 else None,
				"leader_interval_s": self.leader_interval_s,
				"car_interval_s": self.car_interval_s,
			},
		}

	async def status(self) -> Dict[str, Any]:
		state, robot_ids = await self._snapshot_state()
		tasks, tasks_ts = await self._get_tasks_snapshot()
		_, leader_ts = await self._get_leader_snapshot()
		return {
			"side": self.side,
			"started": self._started,
			"robot_count": len(robot_ids),
			"task_count": len(tasks),
			"has_state": bool(state),
			"task_age_s": max(0.0, time.time() - tasks_ts) if tasks_ts > 0 else None,
			"leader_age_s": max(0.0, time.time() - leader_ts) if leader_ts > 0 else None,
			"car_agents": sorted(self._car_agents.keys()),
		}

	async def _leader_loop(self) -> None:
		while not self._stop_event.is_set():
			tick_s = time.time()
			try:
				await self._run_leader_cycle_once()
			except asyncio.CancelledError:
				raise
			except Exception as exc:
				LOGGER.warning("Leader loop error side=%s: %s", self.side, exc)
			await self._sleep_rest(tick_s, self.leader_interval_s)

	async def _car_loop(self) -> None:
		while not self._stop_event.is_set():
			tick_s = time.time()
			try:
				await self._run_car_cycle_once(force=False)
			except asyncio.CancelledError:
				raise
			except Exception as exc:
				LOGGER.warning("Car loop error side=%s: %s", self.side, exc)
			await self._sleep_rest(tick_s, self.car_interval_s)

	async def _run_leader_cycle_once(self) -> Optional[str]:
		async with self._slow_cycle_lock:
			battle_state, _ = await self._snapshot_state()
			if not battle_state:
				return None

			plan = await self.leader_agent.think(global_state=battle_state, side=self.side, force=True)
			async with self._decision_lock:
				self._leader_order_text = str(plan.order_text)
				self._leader_order_ts_s = float(plan.generated_at_s)
			return self._leader_order_text

	async def _run_car_cycle_once(self, force: bool) -> Dict[str, Dict[str, Any]]:
		async with self._fast_cycle_lock:
			battle_state, robot_ids = await self._snapshot_state()
			if not battle_state or not robot_ids:
				if force and robot_ids:
					return _fill_missing_tasks({}, robot_ids)
				return {}

			car_agents = await self._ensure_car_agents(robot_ids)
			leader_order, leader_ts = await self._get_leader_snapshot()
			if (not leader_order.strip()) or (leader_ts <= 0):
				leader_order = "Prioritize survival, maintain spacing, and attack only with advantage."

			local_state_by_robot = _build_local_state_by_robot(
				side=self.side,
				battle_state=battle_state,
				robot_ids=robot_ids,
			)

			tasks = await plan_cars_concurrently(
				car_agents=car_agents,
				local_state_by_robot=local_state_by_robot,
				leader_order=leader_order,
				team_context=battle_state,
				side=self.side,
			)
			tasks = _fill_missing_tasks(tasks, robot_ids)

			async with self._decision_lock:
				self._latest_tasks = copy.deepcopy(tasks)
				self._latest_tasks_ts_s = time.time()

			return tasks

	async def _ensure_car_agents(self, robot_ids: Sequence[str]) -> List[CarAgent]:
		fast_timeout_s = max(0.35, min(2.0, self.car_interval_s * 0.85))
		async with self._agent_lock:
			for robot_id in robot_ids:
				if robot_id in self._car_agents:
					continue
				self._car_agents[robot_id] = CarAgent(
					robot_id=robot_id,
					llm_client=self.llm_client,
					models_cfg=self.models_cfg,
					prompts_cfg=self.prompts_cfg,
					fast_timeout_s=fast_timeout_s,
					reuse_last_task_s=max(1.0, self.car_interval_s * 2.0),
				)

			# Remove long-unused agents when robot list changes.
			active = set(robot_ids)
			stale = [rid for rid in self._car_agents.keys() if rid not in active]
			for rid in stale:
				self._car_agents.pop(rid, None)

			return [self._car_agents[rid] for rid in robot_ids if rid in self._car_agents]

	async def _snapshot_state(self) -> Tuple[Dict[str, Any], List[str]]:
		async with self._state_lock:
			return copy.deepcopy(self._latest_battle_state), list(self._robot_ids)

	async def _get_tasks_snapshot(self) -> Tuple[Dict[str, Dict[str, Any]], float]:
		async with self._decision_lock:
			return copy.deepcopy(self._latest_tasks), float(self._latest_tasks_ts_s)

	async def _get_leader_snapshot(self) -> Tuple[str, float]:
		async with self._decision_lock:
			return str(self._leader_order_text), float(self._leader_order_ts_s)

	async def _sleep_rest(self, tick_start_s: float, interval_s: float) -> None:
		remain_s = float(interval_s) - (time.time() - float(tick_start_s))
		if remain_s <= 0:
			await asyncio.sleep(0)
			return
		try:
			await asyncio.wait_for(self._stop_event.wait(), timeout=remain_s)
		except asyncio.TimeoutError:
			return


class HierarchicalMASManager:
	"""Top-level manager that hosts red/blue side runtimes."""

	def __init__(
		self,
		models_cfg: Mapping[str, Any],
		prompts_cfg: Mapping[str, Any],
		enabled_sides: Sequence[str] = ("red", "blue"),
		ltm_dir: Optional[Path] = None,
	) -> None:
		self.models_cfg = dict(models_cfg)
		self.prompts_cfg = dict(prompts_cfg)

		self.enabled_sides = [s for s in (_normalize_side(x) for x in enabled_sides) if s]
		if not self.enabled_sides:
			raise ValueError("enabled_sides must include red and/or blue")

		base_ltm_dir = Path(ltm_dir) if ltm_dir is not None else (Path(__file__).resolve().parent / "memory" / "data")
		self._llm_clients: Dict[str, AsyncLLMClient] = {}
		self._runtimes: Dict[str, SideMASRuntime] = {}
		for side in self.enabled_sides:
			side_models_cfg = _resolve_side_models_cfg(self.models_cfg, side)
			side_llm_cfg = side_models_cfg.get("llm", {})
			if not isinstance(side_llm_cfg, Mapping):
				side_llm_cfg = {}

			llm_client = AsyncLLMClient.from_models_config(side_models_cfg)
			self._llm_clients[side] = llm_client

			side_path = base_ltm_dir / "ltm_{}.jsonl".format(side)
			self._runtimes[side] = SideMASRuntime(
				side=side,
				llm_client=llm_client,
				models_cfg=side_models_cfg,
				prompts_cfg=self.prompts_cfg,
				ltm_storage_path=side_path,
			)

			leader_cfg = side_models_cfg.get("leader_model", {})
			leader_model_name = ""
			if isinstance(leader_cfg, Mapping):
				leader_model_name = str(leader_cfg.get("name", ""))
			LOGGER.info(
				"Side config resolved: side=%s model=%s base_url=%s api_key_source=%s",
				side,
				leader_model_name,
				side_llm_cfg.get("base_url", ""),
				side_llm_cfg.get("api_key_source", "unknown"),
			)

		self._started = False
		self._start_lock = asyncio.Lock()

	@classmethod
	def from_config_root(
		cls,
		configs_root: Optional[Path] = None,
		enabled_sides: Sequence[str] = ("red", "blue"),
		ltm_dir: Optional[Path] = None,
	) -> "HierarchicalMASManager":
		loader = ConfigLoader(root_dir=configs_root)
		bundle = loader.load_all()
		return cls(
			models_cfg=bundle.models,
			prompts_cfg=bundle.prompts,
			enabled_sides=enabled_sides,
			ltm_dir=ltm_dir,
		)

	async def start(self) -> None:
		async with self._start_lock:
			if self._started:
				return
			for runtime in self._runtimes.values():
				await runtime.start()
			self._started = True
			LOGGER.info("HierarchicalMASManager started for sides=%s", self.enabled_sides)

	async def stop(self) -> None:
		async with self._start_lock:
			if not self._started:
				return
			for runtime in self._runtimes.values():
				await runtime.stop()

			closed_client_ids = set()
			for side in list(self._llm_clients.keys()):
				client = self._llm_clients.get(side)
				if client is None:
					continue
				client_id = id(client)
				if client_id in closed_client_ids:
					continue
				await client.close()
				closed_client_ids.add(client_id)
			self._llm_clients = {}

			self._started = False
			LOGGER.info("HierarchicalMASManager stopped")

	async def handle_plan(self, payload: Optional[Mapping[str, Any]], side_hint: str = "") -> Dict[str, Any]:
		if not self._started:
			await self.start()

		side = _infer_side(payload, side_hint)
		if side not in self._runtimes:
			side = self.enabled_sides[0]

		return await self._runtimes[side].handle_plan_request(payload)

	async def status(self) -> Dict[str, Any]:
		data: Dict[str, Any] = {
			"started": self._started,
			"enabled_sides": list(self.enabled_sides),
			"sides": {},
		}
		for side, runtime in self._runtimes.items():
			data["sides"][side] = await runtime.status()
		return data


def _safe_mapping(value: Any) -> Dict[str, Any]:
	if isinstance(value, Mapping):
		return dict(value)
	return {}


def _normalize_side(value: Any) -> str:
	text = str(value or "").strip().lower()
	if text in ("red", "blue"):
		return text
	return ""


def _as_float(value: Any, default: float) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return float(default)


def _resolve_api_key_for_side(side: str, default_api_key: Any) -> Tuple[str, str]:
	side_key_name = "LLM_API_KEY_{}".format(side.upper())
	side_key = os.getenv(side_key_name, "")
	if side_key:
		return side_key, side_key_name

	shared_key = os.getenv("LLM_API_KEY", "")
	if shared_key:
		return shared_key, "LLM_API_KEY"

	legacy_key = os.getenv("LLM_API", "")
	if legacy_key:
		return legacy_key, "LLM_API"

	sitp_key = os.getenv("SITP_LLM_API_KEY", "")
	if sitp_key:
		return sitp_key, "SITP_LLM_API_KEY"

	config_key = str(default_api_key or "")
	if config_key:
		return config_key, "config"

	return "", "MISSING"


def _resolve_side_models_cfg(models_cfg: Mapping[str, Any], side: str) -> Dict[str, Any]:
	resolved = copy.deepcopy(dict(models_cfg))

	llm_cfg = resolved.get("llm", {})
	if not isinstance(llm_cfg, Mapping):
		llm_cfg = {}
	llm_cfg = dict(llm_cfg)

	api_key, key_source = _resolve_api_key_for_side(side=side, default_api_key=llm_cfg.get("api_key", ""))
	llm_cfg["api_key"] = api_key
	llm_cfg["api_key_source"] = key_source

	resolved["llm"] = llm_cfg
	return resolved


def _extract_battle_state(payload: Mapping[str, Any]) -> Dict[str, Any]:
	raw = payload.get("battle_state", payload)
	if isinstance(raw, Mapping):
		return copy.deepcopy(dict(raw))
	return {"raw_battle_state": raw}


def _extract_robot_ids(payload: Mapping[str, Any], battle_state: Mapping[str, Any]) -> List[str]:
	source = payload.get("robot_ids")
	if not isinstance(source, list):
		source = battle_state.get("my_cars", [])

	ids: List[str] = []
	if isinstance(source, list):
		for item in source:
			if isinstance(item, str) and item.strip():
				ids.append(item.strip())

	if not ids:
		friendly = battle_state.get("friendly", {})
		if isinstance(friendly, Mapping):
			for key in sorted(friendly.keys()):
				if isinstance(key, str) and key.strip():
					ids.append(key.strip())

	# Stable dedupe
	out: List[str] = []
	seen = set()
	for rid in ids:
		if rid in seen:
			continue
		seen.add(rid)
		out.append(rid)
	return out


def _infer_side(payload: Optional[Mapping[str, Any]], side_hint: str = "") -> str:
	p = _safe_mapping(payload)
	side = _normalize_side(p.get("side", ""))
	if side:
		return side

	battle_state = _extract_battle_state(p)
	side = _normalize_side(battle_state.get("team_color", ""))
	if side:
		return side

	robot_ids = _extract_robot_ids(p, battle_state)
	joined = " ".join(robot_ids).lower()
	if "red" in joined:
		return "red"
	if "blue" in joined:
		return "blue"

	return _normalize_side(side_hint)


def _extract_visible_enemies(battle_state: Mapping[str, Any]) -> List[Dict[str, Any]]:
	enemy = battle_state.get("enemy", {})
	if not isinstance(enemy, Mapping):
		return []
	state = enemy.get("state", {})
	if not isinstance(state, Mapping):
		return []

	visible = state.get("visible_enemies")
	if isinstance(visible, list):
		return [dict(v) for v in visible if isinstance(v, Mapping)]

	enemies = state.get("enemies")
	if isinstance(enemies, list):
		return [dict(v) for v in enemies if isinstance(v, Mapping) and v.get("visible", True)]

	if "x" in state and "y" in state and state.get("visible", True):
		return [dict(state)]

	return []


def _build_local_state_by_robot(
	side: str,
	battle_state: Mapping[str, Any],
	robot_ids: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
	friendly = battle_state.get("friendly", {})
	if not isinstance(friendly, Mapping):
		friendly = {}

	visible_enemies = _extract_visible_enemies(battle_state)

	out: Dict[str, Dict[str, Any]] = {}
	for rid in robot_ids:
		friendly_entry = friendly.get(rid, {})
		if not isinstance(friendly_entry, Mapping):
			friendly_entry = {}
		state_map = friendly_entry.get("state", {})
		if not isinstance(state_map, Mapping):
			state_map = {}

		out[rid] = {
			"robot_id": rid,
			"team_color": side,
			"state": dict(state_map),
			"yaw": state_map.get("yaw", 0.0),
			"hp": state_map.get("hp", 100.0),
			"ammo": state_map.get("ammo", 10.0),
			"alive": state_map.get("alive", True),
			"in_combat": state_map.get("in_combat", False),
			"visible_enemies": copy.deepcopy(visible_enemies),
			"safe_point": state_map.get("safe_point", {"x": 0.0, "y": 0.0}),
		}
	return out


def _stop_task(reason: str = "missing robot task") -> Dict[str, Any]:
	return {
		"action": "STOP",
		"target": {"x": 0.0, "y": 0.0,"yaw": 0.0},
		"mode": 0,
		"reason": str(reason),
		"timeout": 1.5,
	}


def _fill_missing_tasks(tasks: Mapping[str, Mapping[str, Any]], robot_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
	out: Dict[str, Dict[str, Any]] = {}
	src = dict(tasks) if isinstance(tasks, Mapping) else {}
	for rid in robot_ids:
		raw = src.get(rid)
		if isinstance(raw, Mapping):
			out[rid] = copy.deepcopy(dict(raw))
		else:
			out[rid] = _stop_task("missing task in fast loop")
	return out


def _default_configs_root() -> Path:
	return Path(__file__).resolve().parent


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Hierarchical MAS Manager")
	parser.add_argument("--configs-root", type=str, default=str(_default_configs_root()), help="MAS root containing configs")
	parser.add_argument("--sides", type=str, default="red,blue", help="Comma-separated sides to enable")
	parser.add_argument("--ltm-dir", type=str, default="", help="Optional LTM storage directory")
	parser.add_argument("--status-interval-s", type=float, default=5.0, help="Print status interval in seconds")
	parser.add_argument("--run-duration-s", type=float, default=0.0, help="Exit after N seconds, 0 means forever")
	parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
	return parser


async def _async_main(args: argparse.Namespace) -> int:
	logging.basicConfig(
		level=getattr(logging, str(args.log_level).upper(), logging.INFO),
		format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
	)

	enabled_sides = [
		s for s in (_normalize_side(item) for item in str(args.sides).split(",")) if s
	]
	if not enabled_sides:
		enabled_sides = ["red", "blue"]

	ltm_dir = Path(args.ltm_dir) if str(args.ltm_dir).strip() else None

	try:
		manager = HierarchicalMASManager.from_config_root(
			configs_root=Path(args.configs_root),
			enabled_sides=enabled_sides,
			ltm_dir=ltm_dir,
		)
	except ConfigError as exc:
		LOGGER.error("Config load failed: %s", exc)
		return 2

	await manager.start()
	start_s = time.time()

	try:
		while True:
			await asyncio.sleep(max(0.5, float(args.status_interval_s)))
			snapshot = await manager.status()
			LOGGER.info("manager status: %s", snapshot)

			if float(args.run_duration_s) > 0:
				if time.time() - start_s >= float(args.run_duration_s):
					break
	finally:
		await manager.stop()

	return 0


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	try:
		code = asyncio.run(_async_main(args))
	except KeyboardInterrupt:
		code = 0
	raise SystemExit(code)


if __name__ == "__main__":
	main()

