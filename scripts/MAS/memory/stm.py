#!/usr/bin/env python3
"""Short-term memory for recent battle-state snapshots."""

from __future__ import annotations

import asyncio
import copy
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class STMEntry:
	timestamp_s: float
	state: Dict[str, Any]
	source: str = "battle_state"
	note: str = ""


class ShortTermMemory:
	"""Recent-memory sliding window used by LeaderAgent."""

	def __init__(self, max_items: int = 12) -> None:
		if int(max_items) <= 0:
			raise ValueError("max_items must be > 0")
		self.max_items = int(max_items)
		self._entries: Deque[STMEntry] = deque(maxlen=self.max_items)
		self._lock = asyncio.Lock()

	async def append(
		self,
		state: Optional[Mapping[str, Any]],
		source: str = "battle_state",
		note: str = "",
		timestamp_s: Optional[float] = None,
	) -> STMEntry:
		snapshot = _safe_state_copy(state)
		entry = STMEntry(
			timestamp_s=float(timestamp_s) if timestamp_s is not None else time.time(),
			state=snapshot,
			source=str(source or "battle_state"),
			note=str(note or ""),
		)
		async with self._lock:
			self._entries.append(entry)
		return entry

	async def extend(
		self,
		states: Sequence[Mapping[str, Any]],
		source: str = "batch",
		timestamp_s: Optional[float] = None,
	) -> int:
		count = 0
		stamp = float(timestamp_s) if timestamp_s is not None else time.time()
		async with self._lock:
			for state in states:
				self._entries.append(
					STMEntry(
						timestamp_s=stamp,
						state=_safe_state_copy(state),
						source=str(source or "batch"),
						note="",
					)
				)
				count += 1
		return count

	async def latest(self) -> Optional[STMEntry]:
		async with self._lock:
			if not self._entries:
				return None
			return self._entries[-1]

	async def recent(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
		async with self._lock:
			snapshot = list(self._entries)

		if limit is not None:
			snapshot = snapshot[-max(0, int(limit)) :]

		return [
			{
				"timestamp_s": item.timestamp_s,
				"source": item.source,
				"note": item.note,
				"state": copy.deepcopy(item.state),
			}
			for item in snapshot
		]

	async def clear(self) -> None:
		async with self._lock:
			self._entries.clear()

	async def size(self) -> int:
		async with self._lock:
			return len(self._entries)

	async def summarize(self, max_lines: int = 8) -> str:
		"""Return concise semantic summary text for prompts."""
		async with self._lock:
			entries = list(self._entries)

		if not entries:
			return "No short-term memory available."

		first = entries[0].state
		last = entries[-1].state
		duration_s = max(0.0, entries[-1].timestamp_s - entries[0].timestamp_s)

		lines: List[str] = []
		lines.append("STM window: {} snapshots over {:.1f}s.".format(len(entries), duration_s))

		friendly_latest = _extract_friendly(last)
		enemy_latest = _extract_enemy(last)
		visible_enemy_count = _count_visible_enemies(enemy_latest)

		lines.append(
			"Latest teams: friendly={} visible_enemies={}.".format(
				len(friendly_latest),
				visible_enemy_count,
			)
		)

		hp_delta_line = _build_hp_delta_line(_extract_friendly(first), friendly_latest)
		if hp_delta_line:
			lines.append(hp_delta_line)

		ammo_line = _build_ammo_line(friendly_latest)
		if ammo_line:
			lines.append(ammo_line)

		note_tail = [item.note for item in entries if item.note]
		if note_tail:
			lines.append("Recent notes: {}".format(" | ".join(note_tail[-3:])))

		if max_lines > 0:
			lines = lines[: int(max_lines)]
		return "\n".join(lines)


def _safe_state_copy(state: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
	if isinstance(state, Mapping):
		return copy.deepcopy(dict(state))
	return {}


def _extract_friendly(state: Mapping[str, Any]) -> Mapping[str, Any]:
	friendly = state.get("friendly", {})
	if isinstance(friendly, Mapping):
		return friendly
	return {}


def _extract_enemy(state: Mapping[str, Any]) -> Mapping[str, Any]:
	enemy = state.get("enemy", {})
	if isinstance(enemy, Mapping):
		return enemy
	return {}


def _extract_robot_state(entry: Any) -> Mapping[str, Any]:
	if not isinstance(entry, Mapping):
		return {}
	state = entry.get("state", {})
	if isinstance(state, Mapping):
		return state
	return {}


def _as_float(value: Any, default: float) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		return float(default)


def _count_visible_enemies(enemy_block: Mapping[str, Any]) -> int:
	if not isinstance(enemy_block, Mapping):
		return 0

	state = enemy_block.get("state", {})
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


def _build_hp_delta_line(friendly_first: Mapping[str, Any], friendly_last: Mapping[str, Any]) -> str:
	robot_ids = sorted(set(list(friendly_first.keys()) + list(friendly_last.keys())))
	if not robot_ids:
		return ""

	chunks: List[str] = []
	for robot_id in robot_ids:
		first_state = _extract_robot_state(friendly_first.get(robot_id, {}))
		last_state = _extract_robot_state(friendly_last.get(robot_id, {}))
		first_hp = _as_float(first_state.get("hp", 100.0), 100.0)
		last_hp = _as_float(last_state.get("hp", first_hp), first_hp)
		delta = last_hp - first_hp
		chunks.append("{}:{:+.1f}".format(robot_id, delta))

	if not chunks:
		return ""
	return "HP delta (first->latest): {}.".format(", ".join(chunks))


def _build_ammo_line(friendly_last: Mapping[str, Any]) -> str:
	if not friendly_last:
		return ""
	chunks: List[str] = []
	for robot_id in sorted(friendly_last.keys()):
		state = _extract_robot_state(friendly_last.get(robot_id, {}))
		ammo = _as_float(state.get("ammo", 0.0), 0.0)
		chunks.append("{}:{:.0f}".format(robot_id, ammo))
	if not chunks:
		return ""
	return "Latest ammo: {}.".format(", ".join(chunks))


__all__ = [
	"STMEntry",
	"ShortTermMemory",
]
