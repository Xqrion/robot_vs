#!/usr/bin/env python3
"""Dual-port non-blocking communication layer for MAS LLM planning.

This server is designed to dock with manager/llm_client.py:
1) It exposes HTTP POST /plan.
2) It can listen on both red(8001) and blue(8002) ports in one process.
3) It uses asyncio + FastAPI handlers for non-blocking IO.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config_loader import ConfigError, ConfigLoader
from mas_manager import HierarchicalMASManager
from llm_api import (
	AsyncLLMClient,
	LLMAPIError,
	LLMResponseFormatError,
	build_messages,
	build_profile_from_models,
	render_prompt,
)


LOGGER = logging.getLogger("mas.llm_server")


def _normalize_side(value: Any) -> str:
	text = str(value or "").strip().lower()
	if text in ("red", "blue"):
		return text
	return ""


def _to_float(value: Any, default: float) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return float(default)


def _to_int(value: Any, default: int) -> int:
	try:
		return int(value)
	except (TypeError, ValueError):
		return int(default)


class MASPlannerService:
	"""Bridge from /plan payload to async LLM planning result."""

	ALLOWED_ACTIONS = {"STOP", "GOTO", "ATTACK"}

	def __init__(self, models_cfg: Mapping[str, Any], prompts_cfg: Mapping[str, Any]) -> None:
		self.models_cfg = dict(models_cfg)
		self.prompts_cfg = dict(prompts_cfg)
		self.client = AsyncLLMClient.from_models_config(self.models_cfg)
		self.car_profile = build_profile_from_models(self.models_cfg, "car_model")

	async def close(self) -> None:
		await self.client.close()

	async def plan(self, payload: Mapping[str, Any], side_hint: str = "") -> Tuple[Dict[str, Any], str, bool]:
		"""Generate normalized tasks.

		Returns:
			(tasks_dict, normalized_side, used_fallback)
		"""
		battle_state = self._extract_battle_state(payload)
		robot_ids = self._extract_robot_ids(payload, battle_state)
		normalized_side = self._infer_side(payload, battle_state, robot_ids, side_hint)

		if not robot_ids:
			return {}, normalized_side, False

		try:
			user_prompt = self._build_user_prompt(payload, battle_state, robot_ids, normalized_side)
			system_prompt = str(self._car_prompt_cfg().get("system_prompt", "")).strip()
			if not system_prompt:
				system_prompt = (
					"You are a multi-robot tactical planner. "
					"Return only JSON actions with robot_id/action/target."
				)

			messages = build_messages(system_prompt=system_prompt, user_prompt=user_prompt)
			actions = await self.client.request_actions(messages=messages, profile=self.car_profile)
			tasks = self._actions_to_task_dict(actions, robot_ids)
			return tasks, normalized_side, False
		except (LLMAPIError, LLMResponseFormatError, ValueError) as exc:
			LOGGER.warning("LLM planning failed, use STOP fallback: %s", exc)
			fallback = self._build_stop_fallback(robot_ids, reason="llm_fallback: {}".format(exc))
			return fallback, normalized_side, True

	def _car_prompt_cfg(self) -> Mapping[str, Any]:
		cfg = self.prompts_cfg.get("car", {})
		if isinstance(cfg, Mapping):
			return cfg
		return {}

	def _build_user_prompt(
		self,
		payload: Mapping[str, Any],
		battle_state: Mapping[str, Any],
		robot_ids: Sequence[str],
		side: str,
	) -> str:
		car_cfg = self._car_prompt_cfg()
		template = str(car_cfg.get("user_template", "")).strip()
		if not template:
			template = (
				"LEADER_ORDER:\n{leader_order}\n\n"
				"CAR_STATE_JSON:\n{car_state}\n\n"
				"TEAM_CONTEXT_JSON:\n{team_context}\n"
			)

		# Provide both current and future placeholders so prompt template can evolve.
		return render_prompt(
			template,
			leader_order=payload.get("leader_order", ""),
			car_state={
				"side": side,
				"robot_ids": list(robot_ids),
				"battle_state": battle_state,
			},
			team_context={
				"side": side,
				"robot_ids": list(robot_ids),
			},
			global_state=battle_state,
			stm_summary=payload.get("stm_summary", ""),
			ltm_summary=payload.get("ltm_summary", ""),
		)

	def _extract_battle_state(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
		raw_state = payload.get("battle_state", payload)
		if isinstance(raw_state, Mapping):
			return raw_state
		return {"raw_battle_state": raw_state}

	def _extract_robot_ids(self, payload: Mapping[str, Any], battle_state: Mapping[str, Any]) -> List[str]:
		raw_ids = payload.get("robot_ids")
		if not isinstance(raw_ids, list):
			raw_ids = battle_state.get("my_cars", [])

		robot_ids: List[str] = []
		if isinstance(raw_ids, list):
			for value in raw_ids:
				if isinstance(value, str) and value.strip():
					robot_ids.append(value.strip())

		if not robot_ids:
			friendly = battle_state.get("friendly", {})
			if isinstance(friendly, Mapping):
				for key in sorted(friendly.keys()):
					if isinstance(key, str) and key.strip():
						robot_ids.append(key.strip())

		# Preserve order but remove duplicates.
		deduped: List[str] = []
		seen = set()
		for robot_id in robot_ids:
			if robot_id in seen:
				continue
			seen.add(robot_id)
			deduped.append(robot_id)
		return deduped

	def _infer_side(
		self,
		payload: Mapping[str, Any],
		battle_state: Mapping[str, Any],
		robot_ids: Sequence[str],
		side_hint: str,
	) -> str:
		side = _normalize_side(payload.get("side", ""))
		if side:
			return side

		side = _normalize_side(battle_state.get("team_color", ""))
		if side:
			return side

		joined = " ".join(robot_ids).lower()
		if "red" in joined:
			return "red"
		if "blue" in joined:
			return "blue"
		return _normalize_side(side_hint)

	def _actions_to_task_dict(self, actions: Sequence[Mapping[str, Any]], robot_ids: Sequence[str]) -> Dict[str, Any]:
		raw_by_robot: Dict[str, Mapping[str, Any]] = {}
		for item in actions:
			if not isinstance(item, Mapping):
				continue
			robot_id = str(item.get("robot_id", "")).strip()
			if not robot_id:
				continue
			raw_by_robot[robot_id] = item

		result: Dict[str, Any] = {}
		for robot_id in robot_ids:
			item = raw_by_robot.get(robot_id)
			if not isinstance(item, Mapping):
				result[robot_id] = self._stop_task("llm missing robot action")
				continue
			result[robot_id] = self._normalize_single_task(item)
		return result

	def _normalize_single_task(self, item: Mapping[str, Any]) -> Dict[str, Any]:
		action = str(item.get("action", "STOP")).strip().upper()
		if action not in self.ALLOWED_ACTIONS:
			action = "STOP"

		target_raw = item.get("target", {})
		if not isinstance(target_raw, Mapping):
			target_raw = {}
		target = {
			"x": _to_float(target_raw.get("x", 0.0), 0.0),
			"y": _to_float(target_raw.get("y", 0.0), 0.0),
		}

		mode_default = 0 if action == "STOP" else (2 if action == "ATTACK" else 1)
		mode = _to_int(item.get("mode", mode_default), mode_default)

		timeout = _to_float(item.get("timeout", 2.0), 2.0)
		timeout = max(0.5, min(30.0, timeout))

		reason = str(item.get("reason", "llm decision")).strip() or "llm decision"

		return {
			"action": action,
			"target": target,
			"mode": mode,
			"reason": reason,
			"timeout": timeout,
		}

	def _stop_task(self, reason: str) -> Dict[str, Any]:
		return {
			"action": "STOP",
			"target": {"x": 0.0, "y": 0.0},
			"mode": 0,
			"reason": str(reason),
			"timeout": 2.0,
		}

	def _build_stop_fallback(self, robot_ids: Sequence[str], reason: str) -> Dict[str, Any]:
		return {robot_id: self._stop_task(reason) for robot_id in robot_ids}


async def _parse_json_payload(request: Request) -> Mapping[str, Any]:
	try:
		payload = await request.json()
	except Exception:
		raw = await request.body()
		text = raw.decode("utf-8", errors="ignore").strip()
		if not text:
			return {}
		try:
			payload = json.loads(text)
		except Exception:
			return {"raw_payload": text}

	if isinstance(payload, Mapping):
		return payload
	return {"raw_payload": payload}


def create_app(manager: HierarchicalMASManager, port_side_map: Mapping[int, str]) -> FastAPI:
	app = FastAPI(title="MAS Dual-Port LLM Server")

	@app.get("/health")
	async def health(request: Request) -> Dict[str, Any]:
		port = int(request.url.port or 0)
		status = await manager.status()
		return {
			"ok": True,
			"port": port,
			"side": port_side_map.get(port, "unknown"),
			"manager": status,
		}

	@app.post("/plan")
	async def plan(request: Request) -> JSONResponse:
		payload = await _parse_json_payload(request)
		local_port = int(request.url.port or 0)
		side_hint = port_side_map.get(local_port, "")

		response_payload = await manager.handle_plan(payload=payload, side_hint=side_hint)
		return JSONResponse(content=response_payload)

	return app


async def run_dual_servers(app: FastAPI, host: str, red_port: int, blue_port: int, log_level: str) -> None:
	red_cfg = uvicorn.Config(
		app=app,
		host=host,
		port=int(red_port),
		log_level=log_level,
		lifespan="off",
	)
	blue_cfg = uvicorn.Config(
		app=app,
		host=host,
		port=int(blue_port),
		log_level=log_level,
		lifespan="off",
	)

	red_server = uvicorn.Server(config=red_cfg)
	blue_server = uvicorn.Server(config=blue_cfg)

	# Uvicorn default signal handling is singleton-oriented. Disable it and
	# let asyncio.run/KeyboardInterrupt control cancellation for both servers.
	red_server.install_signal_handlers = lambda: None
	blue_server.install_signal_handlers = lambda: None

	await asyncio.gather(
		red_server.serve(),
		blue_server.serve(),
	)


def _default_configs_root() -> str:
	return "."


def _build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Dual-port MAS LLM server")
	parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
	parser.add_argument("--red-port", type=int, default=8001, help="Red team listening port")
	parser.add_argument("--blue-port", type=int, default=8002, help="Blue team listening port")
	parser.add_argument(
		"--configs-root",
		type=str,
		default=_default_configs_root(),
		help="Root dir containing scripts/MAS/configs via ConfigLoader root convention",
	)
	parser.add_argument("--log-level", type=str, default="info", help="uvicorn log level")
	return parser


async def _async_main(args: argparse.Namespace) -> int:
	logging.basicConfig(
		level=logging.INFO,
		format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
	)

	try:
		loader = ConfigLoader(root_dir=args.configs_root)
		bundle = loader.load_all()
	except ConfigError as exc:
		LOGGER.error("Load config failed: %s", exc)
		return 2

	runtime_cfg = bundle.models.get("runtime", {})
	if isinstance(runtime_cfg, Mapping):
		port_cfg = runtime_cfg.get("team_ports", {})
		if isinstance(port_cfg, Mapping):
			if int(args.red_port) == 8001:
				args.red_port = _to_int(port_cfg.get("red", 8001), 8001)
			if int(args.blue_port) == 8002:
				args.blue_port = _to_int(port_cfg.get("blue", 8002), 8002)

	manager = HierarchicalMASManager(
		models_cfg=bundle.models,
		prompts_cfg=bundle.prompts,
		enabled_sides=("red", "blue"),
	)
	await manager.start()

	app = create_app(
		manager=manager,
		port_side_map={
			int(args.red_port): "red",
			int(args.blue_port): "blue",
		},
	)

	LOGGER.info(
		"Starting dual-port LLM server host=%s red_port=%d blue_port=%d",
		args.host,
		int(args.red_port),
		int(args.blue_port),
	)

	try:
		await run_dual_servers(
			app=app,
			host=args.host,
			red_port=int(args.red_port),
			blue_port=int(args.blue_port),
			log_level=str(args.log_level),
		)
	finally:
		await manager.stop()

	return 0


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	# By default, root_dir is current script folder. Keep CLI compatibility and
	# allow passing scripts/MAS explicitly when needed.
	if args.configs_root == ".":
		from pathlib import Path

		args.configs_root = str(Path(__file__).resolve().parent)

	try:
		exit_code = asyncio.run(_async_main(args))
	except KeyboardInterrupt:
		LOGGER.info("Received interrupt, shutting down.")
		exit_code = 0
	raise SystemExit(exit_code)


if __name__ == "__main__":
	main()
