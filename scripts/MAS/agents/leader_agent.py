#!/usr/bin/env python3
"""Leader agent: slow strategic reasoning with memory integration."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

try:
	from llm_api import (
		AsyncLLMClient,
		LLMAPIError,
		LLMResponseFormatError,
		build_messages,
		build_profile_from_models,
		render_prompt,
	)
except ImportError:  # pragma: no cover
	from ..llm_api import (  # type: ignore
		AsyncLLMClient,
		LLMAPIError,
		LLMResponseFormatError,
		build_messages,
		build_profile_from_models,
		render_prompt,
	)

try:
	from memory.ltm import LongTermMemory
	from memory.stm import ShortTermMemory
except ImportError:  # pragma: no cover
	from ..memory.ltm import LongTermMemory  # type: ignore
	from ..memory.stm import ShortTermMemory  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LeaderPlan:
	order_text: str
	generated_at_s: float
	side: str = ""
	used_cache: bool = False
	used_fallback: bool = False


class LeaderAgent:
	"""Strategic leader with slow-think behavior and memory-aware prompting."""

	def __init__(
		self,
		llm_client: AsyncLLMClient,
		models_cfg: Mapping[str, Any],
		prompts_cfg: Mapping[str, Any],
		stm: ShortTermMemory,
		ltm: LongTermMemory,
		min_cycle_s: Optional[float] = None,
		max_order_chars: int = 1200,
	) -> None:
		self.llm_client = llm_client
		self.models_cfg = dict(models_cfg)
		self.prompts_cfg = dict(prompts_cfg)
		self.stm = stm
		self.ltm = ltm

		self.profile = build_profile_from_models(self.models_cfg, "leader_model")

		runtime_cfg = self.models_cfg.get("runtime", {})
		runtime_cycle = 5.0
		if isinstance(runtime_cfg, Mapping):
			runtime_cycle = _as_float(runtime_cfg.get("leader_loop_interval_s", 5.0), 5.0)

		self.min_cycle_s = max(0.2, _as_float(min_cycle_s, runtime_cycle) if min_cycle_s is not None else runtime_cycle)
		self.max_order_chars = max(256, int(max_order_chars))

		self._lock = asyncio.Lock()
		self._last_plan: Optional[LeaderPlan] = None

	async def observe(self, global_state: Optional[Mapping[str, Any]], note: str = "") -> None:
		await self.stm.append(state=global_state, source="leader_observe", note=note)

	async def think(
		self,
		global_state: Optional[Mapping[str, Any]],
		side: str = "",
		force: bool = False,
	) -> LeaderPlan:
		"""Return a strategic order string with slow-cycle cache semantics."""
		safe_state = _safe_mapping(global_state)
		normalized_side = _normalize_side(side) or _normalize_side(safe_state.get("team_color", ""))

		await self.observe(safe_state, note="leader_cycle")

		async with self._lock:
			now_s = time.time()
			if not force and self._last_plan is not None:
				if now_s - self._last_plan.generated_at_s < self.min_cycle_s:
					cached = self._last_plan
					return LeaderPlan(
						order_text=cached.order_text,
						generated_at_s=cached.generated_at_s,
						side=cached.side,
						used_cache=True,
						used_fallback=cached.used_fallback,
					)

			stm_summary = await self.stm.summarize(max_lines=8)
			ltm_summary = await self.ltm.summarize(limit=6, tags=[normalized_side] if normalized_side else None)

			used_fallback = False
			try:
				messages = self._build_messages(
					global_state=safe_state,
					stm_summary=stm_summary,
					ltm_summary=ltm_summary,
				)
				raw_text = await self.llm_client.request_text(
					messages=messages,
					profile=self.profile,
					trace_tag="leader:{}".format(normalized_side or "unknown"),
				)
			except (LLMAPIError, LLMResponseFormatError, ValueError) as exc:
				LOGGER.warning("LeaderAgent LLM plan failed, using fallback strategy: %s", exc)
				raw_text = self._fallback_strategy_text(safe_state, stm_summary)
				used_fallback = True

			normalized_text = self._normalize_order_text(raw_text)

			# LOGGER.info("LeaderAgent(side=%s) order=%s", normalized_side or "unknown", normalized_text)
			
			plan = LeaderPlan(
				order_text=normalized_text,
				generated_at_s=now_s,
				side=normalized_side,
				used_cache=False,
				used_fallback=used_fallback,
			)
			self._last_plan = plan

		# Persist compact strategic trace to LTM outside lock.
		tags = ["leader", "strategy"]
		if normalized_side:
			tags.append(normalized_side)
		try:
			await self.ltm.add_record(
				record_type="leader_order",
				summary=_truncate(normalized_text, 220),
				payload={
					"side": normalized_side,
					"used_fallback": used_fallback,
				},
				tags=tags,
				score=0.8,
				persist=True,
			)
		except Exception as exc:  # pragma: no cover
			LOGGER.warning("LeaderAgent failed to persist LTM record: %s", exc)

		return plan

	async def get_cached_plan(self) -> Optional[LeaderPlan]:
		async with self._lock:
			return self._last_plan

	def _leader_prompt_cfg(self) -> Mapping[str, Any]:
		cfg = self.prompts_cfg.get("leader", {})
		if isinstance(cfg, Mapping):
			return cfg
		return {}

	def _build_messages(self, global_state: Mapping[str, Any], stm_summary: str, ltm_summary: str) -> Sequence[Dict[str, str]]:
		cfg = self._leader_prompt_cfg()
		system_prompt = str(cfg.get("system_prompt", "")).strip()
		if not system_prompt:
			system_prompt = (
				"You are a multi-robot strategic commander. "
				"Return concise tactical guidance in plain text."
			)

		template = str(cfg.get("user_template", "")).strip()
		if not template:
			template = (
				"GLOBAL_STATE_JSON:\n{global_state}\n\n"
				"SHORT_TERM_MEMORY:\n{stm_summary}\n\n"
				"LONG_TERM_MEMORY:\n{ltm_summary}\n"
			)

		user_prompt = render_prompt(
			template,
			global_state=global_state,
			stm_summary=stm_summary,
			ltm_summary=ltm_summary,
		)
		return build_messages(system_prompt=system_prompt, user_prompt=user_prompt)

	def _normalize_order_text(self, raw_text: Any) -> str:
		text = str(raw_text or "").strip()
		text = _strip_code_fence(text)

		max_lines = 12
		cfg = self._leader_prompt_cfg()
		contract = cfg.get("output_contract", {})
		if isinstance(contract, Mapping):
			max_lines = max(1, _as_int(contract.get("max_lines", 12), 12))

		lines = [line.rstrip() for line in text.splitlines() if line.strip()]
		if not lines:
			lines = ["Hold formation, maintain spacing, and preserve survivability."]
		lines = lines[:max_lines]

		merged = "\n".join(lines)
		return _truncate(merged, self.max_order_chars)

	def _fallback_strategy_text(self, global_state: Mapping[str, Any], stm_summary: str) -> str:
		visible = _visible_enemy_count(global_state)
		friendly = _safe_mapping(global_state.get("friendly", {}))
		alive_count = 0
		low_hp_count = 0

		for value in friendly.values():
			state = _extract_nested_state(value)
			alive = bool(state.get("alive", True))
			hp = _as_float(state.get("hp", 100.0), 100.0)
			if alive and hp > 0:
				alive_count += 1
			if hp < 25.0:
				low_hp_count += 1

		if visible > 0 and low_hp_count == 0:
			return (
				"1) Focus fire on visible enemies while keeping two-angle pressure.\n"
				"2) Keep one robot anchoring and one robot flanking.\n"
				"3) Re-evaluate in next slow cycle."
			)

		if low_hp_count > 0:
			return (
				"1) Prioritize survival: low-HP units disengage to safer corridors.\n"
				"2) High-HP unit screens and delays enemy advance.\n"
				"3) Avoid over-commit until regroup complete."
			)

		if alive_count <= 1:
			return (
				"1) Preserve last active unit and avoid direct duel.\n"
				"2) Use short peeks and opportunistic attacks only.\n"
				"3) Keep movement unpredictable."
			)

		return (
			"1) Keep patrol pressure on key lanes.\n"
			"2) Maintain crossfire spacing and avoid stacking.\n"
			"3) Update tactical focus from next STM snapshot.\n"
			"STM hint: {}".format(_truncate(stm_summary.replace("\n", " | "), 160))
		)


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


def _strip_code_fence(text: str) -> str:
	stripped = text.strip()
	if not stripped.startswith("```"):
		return stripped
	lines = stripped.splitlines()
	if len(lines) >= 2 and lines[-1].strip() == "```":
		return "\n".join(lines[1:-1]).strip()
	return stripped


def _truncate(text: str, max_chars: int) -> str:
	if len(text) <= max_chars:
		return text
	if max_chars <= 3:
		return text[:max_chars]
	return text[: max_chars - 3] + "..."


def _extract_nested_state(value: Any) -> Mapping[str, Any]:
	if isinstance(value, Mapping):
		state = value.get("state", {})
		if isinstance(state, Mapping):
			return state
	return {}


def _visible_enemy_count(global_state: Mapping[str, Any]) -> int:
	enemy = global_state.get("enemy", {})
	if not isinstance(enemy, Mapping):
		return 0
	state = enemy.get("state", {})
	if not isinstance(state, Mapping):
		return 0

	visible = state.get("visible_enemies")
	if isinstance(visible, list):
		return len([item for item in visible if isinstance(item, Mapping)])

	enemies = state.get("enemies")
	if isinstance(enemies, list):
		return len([item for item in enemies if isinstance(item, Mapping) and item.get("visible", True)])

	if "x" in state and "y" in state and state.get("visible", True):
		return 1
	return 0


__all__ = [
	"LeaderAgent",
	"LeaderPlan",
]
