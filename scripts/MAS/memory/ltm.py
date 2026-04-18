#!/usr/bin/env python3
"""Long-term memory store for cross-match tactical experience."""

from __future__ import annotations

import asyncio
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class LTMRecord:
	timestamp_s: float
	record_type: str
	summary: str
	payload: Dict[str, Any]
	tags: List[str]
	score: float = 1.0

	def to_dict(self) -> Dict[str, Any]:
		return {
			"timestamp_s": float(self.timestamp_s),
			"record_type": str(self.record_type),
			"summary": str(self.summary),
			"payload": copy.deepcopy(self.payload),
			"tags": list(self.tags),
			"score": float(self.score),
		}


class LongTermMemory:
	"""Persistent memory backed by JSONL for low-overhead append operations."""

	def __init__(self, storage_path: Optional[Path] = None, max_in_memory: int = 2000) -> None:
		base_dir = Path(__file__).resolve().parent
		default_path = base_dir / "data" / "ltm_records.jsonl"
		self.storage_path = Path(storage_path) if storage_path is not None else default_path
		self.max_in_memory = max(1, int(max_in_memory))

		self._records: List[LTMRecord] = []
		self._loaded = False
		self._lock = asyncio.Lock()

	async def ensure_loaded(self) -> None:
		async with self._lock:
			if self._loaded:
				return

			records = await asyncio.to_thread(_read_records_from_disk, self.storage_path)
			self._records = records[-self.max_in_memory :]
			self._loaded = True

	async def add_record(
		self,
		record_type: str,
		summary: str,
		payload: Optional[Mapping[str, Any]] = None,
		tags: Optional[Sequence[str]] = None,
		score: float = 1.0,
		timestamp_s: Optional[float] = None,
		persist: bool = True,
	) -> LTMRecord:
		if not str(record_type).strip():
			raise ValueError("record_type must not be empty")
		if not str(summary).strip():
			raise ValueError("summary must not be empty")

		await self.ensure_loaded()
		record = LTMRecord(
			timestamp_s=float(timestamp_s) if timestamp_s is not None else time.time(),
			record_type=str(record_type).strip(),
			summary=str(summary).strip(),
			payload=_safe_payload_copy(payload),
			tags=_normalize_tags(tags),
			score=float(score),
		)

		async with self._lock:
			self._records.append(record)
			if len(self._records) > self.max_in_memory:
				self._records = self._records[-self.max_in_memory :]

		if persist:
			await asyncio.to_thread(_append_record_to_disk, self.storage_path, record)
		return record

	async def recent(
		self,
		limit: int = 20,
		record_type: str = "",
		tags: Optional[Sequence[str]] = None,
	) -> List[Dict[str, Any]]:
		await self.ensure_loaded()

		target_type = str(record_type or "").strip()
		target_tags = set(_normalize_tags(tags))

		async with self._lock:
			data = list(self._records)

		if target_type:
			data = [item for item in data if item.record_type == target_type]

		if target_tags:
			data = [item for item in data if target_tags.intersection(set(item.tags))]

		data = data[-max(0, int(limit)) :]
		return [item.to_dict() for item in data]

	async def summarize(
		self,
		limit: int = 6,
		tags: Optional[Sequence[str]] = None,
		record_type: str = "",
	) -> str:
		await self.ensure_loaded()
		target_tags = set(_normalize_tags(tags))
		target_type = str(record_type or "").strip()

		async with self._lock:
			data = list(self._records)

		if target_type:
			data = [item for item in data if item.record_type == target_type]
		if target_tags:
			data = [item for item in data if target_tags.intersection(set(item.tags))]

		if not data:
			return "No long-term memory available."

		# Keep the highest-priority insights while preserving recency preference.
		data_sorted = sorted(data, key=lambda x: (x.score, x.timestamp_s), reverse=True)
		selected = data_sorted[: max(1, int(limit))]

		lines: List[str] = ["LTM selected insights: {} entries.".format(len(selected))]
		for item in selected:
			tag_text = ",".join(item.tags[:4]) if item.tags else "none"
			lines.append("[{}|{:.2f}] {} (tags={})".format(item.record_type, item.score, item.summary, tag_text))
		return "\n".join(lines)

	async def save_lessons(self, lessons_text: str, tags: Optional[Sequence[str]] = None, score: float = 1.0) -> int:
		"""Split plain-text lessons by line and store as separate LTM records."""
		text = str(lessons_text or "").strip()
		if not text:
			return 0

		added = 0
		for raw_line in text.splitlines():
			line = raw_line.strip(" -\t")
			if not line:
				continue
			await self.add_record(
				record_type="lesson",
				summary=line,
				payload={},
				tags=tags,
				score=score,
				persist=True,
			)
			added += 1
		return added

	async def clear(self, persist: bool = False) -> None:
		await self.ensure_loaded()
		async with self._lock:
			self._records = []
		if persist:
			await asyncio.to_thread(_rewrite_records_on_disk, self.storage_path, [])


def _safe_payload_copy(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
	if isinstance(payload, Mapping):
		return copy.deepcopy(dict(payload))
	return {}


def _normalize_tags(tags: Optional[Sequence[str]]) -> List[str]:
	if not tags:
		return []
	normalized: List[str] = []
	seen = set()
	for value in tags:
		if not isinstance(value, str):
			continue
		item = value.strip().lower()
		if not item or item in seen:
			continue
		seen.add(item)
		normalized.append(item)
	return normalized


def _read_records_from_disk(path: Path) -> List[LTMRecord]:
	if not path.exists() or not path.is_file():
		return []

	records: List[LTMRecord] = []
	try:
		with path.open("r", encoding="utf-8") as f:
			for line in f:
				line = line.strip()
				if not line:
					continue
				try:
					obj = json.loads(line)
				except Exception:
					continue

				if not isinstance(obj, Mapping):
					continue

				summary = str(obj.get("summary", "")).strip()
				record_type = str(obj.get("record_type", "")).strip()
				if not summary or not record_type:
					continue

				records.append(
					LTMRecord(
						timestamp_s=float(obj.get("timestamp_s", time.time())),
						record_type=record_type,
						summary=summary,
						payload=_safe_payload_copy(obj.get("payload", {})),
						tags=_normalize_tags(obj.get("tags", [])),
						score=float(obj.get("score", 1.0)),
					)
				)
	except Exception:
		return records

	return records


def _append_record_to_disk(path: Path, record: LTMRecord) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(record.to_dict(), ensure_ascii=False))
		f.write("\n")


def _rewrite_records_on_disk(path: Path, records: Sequence[LTMRecord]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		for item in records:
			f.write(json.dumps(item.to_dict(), ensure_ascii=False))
			f.write("\n")


__all__ = [
	"LTMRecord",
	"LongTermMemory",
]
