#!/usr/bin/env python3
"""Car agent: fast tactical execution with per-robot independent requests."""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
	from llm_api import (
		AsyncLLMClient,
		LLMAPIError,
		LLMRequestProfile,
		LLMResponseFormatError,
		build_messages,
		build_profile_from_models,
		render_prompt,
	)
except ImportError:  # pragma: no cover
	from ..llm_api import (  # type: ignore
		AsyncLLMClient,
		LLMAPIError,
		LLMRequestProfile,
		LLMResponseFormatError,
		build_messages,
		build_profile_from_models,
		render_prompt,
	)


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CarDecision:
	robot_id: str
	task: Dict[str, Any]
	generated_at_s: float
	used_fallback: bool = False


class CarAgent:
	"""Fast tactical agent for one robot.

	Fast characteristics:
	  - low-latency timeout-bound inference,
	  - zero extra retry by default,
	  - immediate fallback when LLM call fails.
	"""

	ALLOWED_ACTIONS = {"STOP", "GOTO", "ATTACK"}

	def __init__(
		self,
		robot_id: str,
		llm_client: AsyncLLMClient,
		models_cfg: Mapping[str, Any],
		prompts_cfg: Mapping[str, Any],
		fast_timeout_s: Optional[float] = None,
		reuse_last_task_s: float = 1.5,
	) -> None:
		rid = str(robot_id or "").strip()
		if not rid:
			raise ValueError("robot_id must not be empty")

		self.robot_id = rid
		self.llm_client = llm_client
		self.models_cfg = dict(models_cfg)
		self.prompts_cfg = dict(prompts_cfg)

		base_profile = build_profile_from_models(self.models_cfg, "car_model")
		target_timeout = max(0.3, _as_float(fast_timeout_s, base_profile.timeout_s) if fast_timeout_s is not None else base_profile.timeout_s)

		# Fast profile intentionally trims retries to avoid blocking the 1s loop.
		self.fast_profile = LLMRequestProfile(
			model=base_profile.model,
			temperature=base_profile.temperature,
			max_tokens=base_profile.max_tokens,
			top_p=base_profile.top_p,
			timeout_s=min(base_profile.timeout_s, target_timeout),
			retries=0,
			backoff_s=min(base_profile.backoff_s, 0.1),
		)

		self.reuse_last_task_s = max(0.0, float(reuse_last_task_s))
		self._lock = asyncio.Lock()
		self._last_task: Dict[str, Any] = self._stop_task("init")
		self._last_task_time_s = 0.0

	async def act(
		self,
		local_state: Optional[Mapping[str, Any]],
		leader_order: str,
		team_context: Optional[Mapping[str, Any]] = None,
		side: str = "",
	) -> CarDecision:
		safe_local = _safe_mapping(local_state)
		safe_team_context = _safe_mapping(team_context)
		now_s = time.time()

		try:
			messages = self._build_messages(
				leader_order=leader_order,
				local_state=safe_local,
				team_context=safe_team_context,
				side=side,
			)
			actions = await asyncio.wait_for(
				self.llm_client.request_actions(
					messages=messages,
					profile=self.fast_profile,
					trace_tag="car:{}".format(self.robot_id),
				),
				timeout=max(0.35, self.fast_profile.timeout_s + 0.05),
			)
			task = self._pick_and_normalize_task(actions)
			used_fallback = False
		except (asyncio.TimeoutError, LLMAPIError, LLMResponseFormatError, ValueError) as exc:
			LOGGER.warning("CarAgent(%s) LLM action failed, using fast fallback: %s", self.robot_id, exc)
			last_task = await self._get_last_task_if_recent(now_s)
			if last_task is not None:
				task = copy.deepcopy(last_task)
				task["reason"] = "reuse_last_task_after_error"
				used_fallback = True
			else:
				task = self._rule_fallback_task(safe_local, safe_team_context, reason=str(exc))
				used_fallback = True

		async with self._lock:
			self._last_task = copy.deepcopy(task)
			self._last_task_time_s = now_s

		return CarDecision(
			robot_id=self.robot_id,
			task=task,
			generated_at_s=now_s,
			used_fallback=used_fallback,
		)

	async def get_last_task(self) -> Dict[str, Any]:
		async with self._lock:
			return copy.deepcopy(self._last_task)

	async def emergency_task(self, reason: str = "emergency") -> Dict[str, Any]:
		now_s = time.time()
		last_task = await self._get_last_task_if_recent(now_s)
		if last_task is not None:
			task = copy.deepcopy(last_task)
			task["reason"] = "emergency_reuse_last_task"
			return task
		return self._stop_task(reason)

	def _car_prompt_cfg(self) -> Mapping[str, Any]:
		cfg = self.prompts_cfg.get("car", {})
		if isinstance(cfg, Mapping):
			return cfg
		return {}

	def _build_messages(
		self,
		leader_order: str,
		local_state: Mapping[str, Any],
		team_context: Mapping[str, Any],
		side: str,
	) -> Sequence[Dict[str, str]]:
		cfg = self._car_prompt_cfg()
		system_prompt = str(cfg.get("system_prompt", "")).strip()
		if not system_prompt:
			system_prompt = (
				"You are a robot tactical executor. "
				"Return a JSON array of actions only."
			)

		template = str(cfg.get("user_template", "")).strip()
		if not template:
			template = (
				"LEADER_ORDER:\n{leader_order}\n\n"
				"CAR_STATE_JSON:\n{car_state}\n\n"
				"TEAM_CONTEXT_JSON:\n{team_context}\n"
			)

		user_prompt = render_prompt(
			template,
			leader_order=str(leader_order or ""),
			car_state={
				"robot_id": self.robot_id,
				"side": _normalize_side(side),
				"local_state": local_state,
			},
			team_context=team_context,
			global_state=team_context,
			stm_summary="",
			ltm_summary="",
		)
		return build_messages(system_prompt=system_prompt, user_prompt=user_prompt)

	def _pick_and_normalize_task(self, actions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
		exact: Optional[Mapping[str, Any]] = None
		first: Optional[Mapping[str, Any]] = None

		for item in actions:
			if not isinstance(item, Mapping):
				continue
			if first is None:
				first = item
			rid = str(item.get("robot_id", "")).strip()
			if rid == self.robot_id:
				exact = item
				break

		selected = exact or first
		if not isinstance(selected, Mapping):
			return self._stop_task("empty llm action")
		return self._normalize_task(selected)

	def _normalize_task(self, raw: Mapping[str, Any]) -> Dict[str, Any]:
		action = str(raw.get("action", raw.get("cmd", raw.get("type", "STOP")))).strip().upper()
		action = _normalize_action_alias(action)
		if action not in self.ALLOWED_ACTIONS:
			action = "STOP"

		target_raw = raw.get("target", {})
		if not isinstance(target_raw, Mapping):
			target_raw = {}
		target = {
			"x": _as_float(target_raw.get("x", 0.0), 0.0),
			"y": _as_float(target_raw.get("y", 0.0), 0.0),
		}

		mode_default = 0 if action == "STOP" else (2 if action == "ATTACK" else 1)
		mode = _as_int(raw.get("mode", mode_default), mode_default)

		timeout_s = _as_float(raw.get("timeout", 2.0), 2.0)
		timeout_s = max(0.5, min(15.0, timeout_s))

		reason = str(raw.get("reason", "car llm decision")).strip() or "car llm decision"
		reason = _truncate(reason, 180)

		return {
			"action": action,
			"target": target,
			"mode": mode,
			"reason": reason,
			"timeout": timeout_s,
		}

	async def _get_last_task_if_recent(self, now_s: float) -> Optional[Dict[str, Any]]:
		async with self._lock:
			if self._last_task_time_s <= 0:
				return None
			if now_s - self._last_task_time_s > self.reuse_last_task_s:
				return None
			return copy.deepcopy(self._last_task)

	def _rule_fallback_task(
		self,
		local_state: Mapping[str, Any],
		team_context: Mapping[str, Any],
		reason: str,
	) -> Dict[str, Any]:
		enemy_point = _extract_enemy_point(local_state)
		if enemy_point is None:
			enemy_point = _extract_enemy_point(team_context)

		hp = _as_float(_read_local_value(local_state, "hp", 100.0), 100.0)
		ammo = _as_float(_read_local_value(local_state, "ammo", 10.0), 10.0)

		if enemy_point is not None and ammo > 0 and hp > 15.0:
			return {
				"action": "ATTACK",
				"target": enemy_point,
				"mode": 2,
				"reason": _truncate("fallback_attack_visible_enemy: {}".format(reason), 180),
				"timeout": 1.5,
			}

		if hp <= 15.0 or ammo <= 0.0:
			safe_point = _extract_safe_point(local_state)
			return {
				"action": "GOTO",
				"target": safe_point,
				"mode": 3,
				"reason": _truncate("fallback_retreat: {}".format(reason), 180),
				"timeout": 2.5,
			}

		return self._stop_task("fallback_stop: {}".format(_truncate(reason, 120)))

	def _stop_task(self, reason: str) -> Dict[str, Any]:
		return {
			"action": "STOP",
			"target": {"x": 0.0, "y": 0.0},
			"mode": 0,
			"reason": str(reason),
			"timeout": 1.5,
		}


async def plan_cars_concurrently(
	car_agents: Sequence[CarAgent],
	local_state_by_robot: Mapping[str, Mapping[str, Any]],
	leader_order: str,
	team_context: Optional[Mapping[str, Any]] = None,
	side: str = "",
) -> Dict[str, Dict[str, Any]]:
	"""Run all CarAgent calls concurrently (independent per-robot LLM requests)."""
	shared_context = _safe_mapping(team_context)
	coroutines = []
	for agent in car_agents:
		local_state = _safe_mapping(local_state_by_robot.get(agent.robot_id, {}))
		coroutines.append(
			agent.act(
				local_state=local_state,
				leader_order=leader_order,
				team_context=shared_context,
				side=side,
			)
		)

	results = await asyncio.gather(*coroutines, return_exceptions=True)

	merged: Dict[str, Dict[str, Any]] = {}
	for agent, result in zip(car_agents, results):
		if isinstance(result, Exception):
			LOGGER.warning("CarAgent(%s) gather result exception: %s", agent.robot_id, result)
			merged[agent.robot_id] = await agent.emergency_task(reason="gather_exception")
			continue
		merged[result.robot_id] = result.task
	return merged


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


def _as_int(value: Any, default: int) -> int:
	try:
		return int(value)
	except (TypeError, ValueError):
		return int(default)


def _truncate(text: str, max_chars: int) -> str:
	if len(text) <= max_chars:
		return text
	if max_chars <= 3:
		return text[:max_chars]
	return text[: max_chars - 3] + "..."


def _extract_enemy_point(source: Mapping[str, Any]) -> Optional[Dict[str, float]]:
	# Format A: local_state.visible_enemies = [{x, y}, ...]
	visible = source.get("visible_enemies")
	if isinstance(visible, list):
		for item in visible:
			if isinstance(item, Mapping) and "x" in item and "y" in item:
				return {
					"x": _as_float(item.get("x", 0.0), 0.0),
					"y": _as_float(item.get("y", 0.0), 0.0),
				}

	# Format B: source.enemy.state.visible_enemies = [...]
	enemy = source.get("enemy", {})
	if isinstance(enemy, Mapping):
		state = enemy.get("state", {})
		if isinstance(state, Mapping):
			visible2 = state.get("visible_enemies")
			if isinstance(visible2, list):
				for item in visible2:
					if isinstance(item, Mapping) and "x" in item and "y" in item:
						return {
							"x": _as_float(item.get("x", 0.0), 0.0),
							"y": _as_float(item.get("y", 0.0), 0.0),
						}

			enemies = state.get("enemies")
			if isinstance(enemies, list):
				for item in enemies:
					if not isinstance(item, Mapping):
						continue
					if not item.get("visible", True):
						continue
					if "x" in item and "y" in item:
						return {
							"x": _as_float(item.get("x", 0.0), 0.0),
							"y": _as_float(item.get("y", 0.0), 0.0),
						}

	return None


def _read_local_value(local_state: Mapping[str, Any], key: str, default: Any) -> Any:
	if key in local_state:
		return local_state.get(key, default)

	nested = local_state.get("state", {})
	if isinstance(nested, Mapping):
		return nested.get(key, default)
	return default


def _extract_safe_point(local_state: Mapping[str, Any]) -> Dict[str, float]:
	for key in ("safe_point", "fallback_point", "home_point"):
		point = local_state.get(key)
		if isinstance(point, Mapping):
			return {
				"x": _as_float(point.get("x", 0.0), 0.0),
				"y": _as_float(point.get("y", 0.0), 0.0),
			}
	return {"x": 0.0, "y": 0.0}


def _normalize_action_alias(action: str) -> str:
	value = str(action or "").strip().upper()
	if not value:
		return "STOP"

	alias = {
		"MOVE": "GOTO",
		"GO": "GOTO",
		"GO_TO": "GOTO",
		"NAV": "GOTO",
		"NAVIGATE": "GOTO",
		"PATROL": "GOTO",
		"FIRE": "ATTACK",
		"SHOOT": "ATTACK",
		"ENGAGE": "ATTACK",
		"IDLE": "STOP",
		"HOLD": "STOP",
		"WAIT": "STOP",
	}
	return alias.get(value, value)


__all__ = [
	"CarAgent",
	"CarDecision",
	"plan_cars_concurrently",
]
