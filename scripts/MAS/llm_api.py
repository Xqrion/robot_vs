#!/usr/bin/env python3
"""Async LLM client utilities for the hierarchical MAS pipeline.

Design goals:
1) pure asyncio interface,
2) independent concurrent requests per CarAgent,
3) robust retries for transient failures,
4) tolerant parsing of JSON action outputs.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx

try:
	from openai import AsyncOpenAI
except Exception:  # pragma: no cover
	AsyncOpenAI = None  # type: ignore


logger = logging.getLogger(__name__)

_RE_JSON_ARRAY = re.compile(r"\[[\s\S]*\]")
_RE_JSON_OBJECT = re.compile(r"\{[\s\S]*\}")


class LLMAPIError(RuntimeError):
	"""Raised when upstream LLM API invocation fails."""


class LLMResponseFormatError(LLMAPIError):
	"""Raised when model output does not match expected output contract."""


@dataclass(frozen=True)
class LLMRequestProfile:
	model: str
	temperature: float = 0.2
	max_tokens: int = 256
	top_p: float = 1.0
	timeout_s: float = 8.0
	retries: int = 2
	backoff_s: float = 0.4


class _RetriableStatusError(Exception):
	def __init__(self, status_code: int, detail: str) -> None:
		super(_RetriableStatusError, self).__init__("HTTP {}: {}".format(status_code, detail))
		self.status_code = status_code
		self.detail = detail


def _as_float(value: Any, field_name: str) -> float:
	try:
		return float(value)
	except (TypeError, ValueError):
		raise ValueError("{} must be float-like, got {}".format(field_name, value))


def _as_int(value: Any, field_name: str) -> int:
	try:
		return int(value)
	except (TypeError, ValueError):
		raise ValueError("{} must be int-like, got {}".format(field_name, value))


def _as_bool(value: Any, default: bool = False) -> bool:
	if isinstance(value, bool):
		return value
	if value is None:
		return bool(default)

	text = str(value).strip().lower()
	if text in ("1", "true", "yes", "y", "on"):
		return True
	if text in ("0", "false", "no", "n", "off"):
		return False
	return bool(default)


def _utc_now_iso() -> str:
	return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _single_line_preview(text: Any, max_chars: int = 220) -> str:
	one_line = " ".join(str(text or "").split())
	if len(one_line) <= max_chars:
		return one_line
	if max_chars <= 3:
		return one_line[:max_chars]
	return one_line[: max_chars - 3] + "..."


def build_profile_from_models(models_cfg: Mapping[str, Any], section_name: str) -> LLMRequestProfile:
	"""Build LLM request profile from models.yaml sections.

	Args:
		models_cfg: Loaded models config dictionary.
		section_name: Usually "leader_model" or "car_model".
	"""
	if not isinstance(models_cfg, Mapping):
		raise ValueError("models_cfg must be a mapping")

	llm_cfg = models_cfg.get("llm", {})
	model_cfg = models_cfg.get(section_name, {})
	if not isinstance(llm_cfg, Mapping):
		llm_cfg = {}
	if not isinstance(model_cfg, Mapping):
		model_cfg = {}

	model_name = str(model_cfg.get("name", "")).strip()
	if not model_name:
		raise ValueError("{}.name is required in models config".format(section_name))

	return LLMRequestProfile(
		model=model_name,
		temperature=_as_float(model_cfg.get("temperature", 0.2), "{}.temperature".format(section_name)),
		max_tokens=_as_int(model_cfg.get("max_tokens", 256), "{}.max_tokens".format(section_name)),
		top_p=_as_float(model_cfg.get("top_p", 1.0), "{}.top_p".format(section_name)),
		timeout_s=_as_float(
			model_cfg.get("timeout_s", llm_cfg.get("default_timeout_s", 8.0)),
			"{}.timeout_s".format(section_name),
		),
		retries=_as_int(
			model_cfg.get("retries", llm_cfg.get("default_retries", 2)),
			"{}.retries".format(section_name),
		),
		backoff_s=_as_float(
			model_cfg.get("backoff_s", llm_cfg.get("default_backoff_s", 0.4)),
			"{}.backoff_s".format(section_name),
		),
	)


def render_prompt(template: str, **kwargs: Any) -> str:
	"""Render user prompt template with dict/list values serialized to JSON."""
	normalized: Dict[str, str] = {}
	for key, value in kwargs.items():
		if isinstance(value, str):
			normalized[key] = value
		else:
			normalized[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
	try:
		return template.format(**normalized)
	except KeyError as exc:
		raise ValueError("Prompt template is missing placeholder: {}".format(exc))


def build_messages(system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
	return [
		{"role": "system", "content": str(system_prompt)},
		{"role": "user", "content": str(user_prompt)},
	]


def _strip_code_fence(text: str) -> str:
	stripped = text.strip()
	if not stripped.startswith("```"):
		return stripped

	lines = stripped.splitlines()
	if len(lines) >= 2 and lines[-1].strip() == "```":
		body = lines[1:-1]
		return "\n".join(body).strip()
	return stripped


def _json_loads_tolerant(raw_text: str) -> Any:
	text = _strip_code_fence(raw_text)

	try:
		return json.loads(text)
	except Exception:
		pass

	match_array = _RE_JSON_ARRAY.search(text)
	if match_array:
		try:
			return json.loads(match_array.group(0))
		except Exception:
			pass

	match_object = _RE_JSON_OBJECT.search(text)
	if match_object:
		try:
			return json.loads(match_object.group(0))
		except Exception:
			pass

	raise LLMResponseFormatError("LLM output does not contain valid JSON")


def parse_action_list(raw_output: Any) -> List[Dict[str, Any]]:
	"""Parse model output into canonical action list format.

	Canonical item shape:
	  {"robot_id": str(optional for single-robot), "action": str, "target": Any, ...optional fields...}
	"""
	parsed = raw_output
	if isinstance(raw_output, str):
		parsed = _json_loads_tolerant(raw_output)

	action_items: List[Any]
	if isinstance(parsed, list):
		action_items = parsed
	elif isinstance(parsed, Mapping):
		for key in ("actions", "commands", "result", "data"):
			candidate = parsed.get(key)
			if isinstance(candidate, list):
				action_items = candidate
				break
			if key in ("result", "data") and isinstance(candidate, Mapping):
				# Support {"result": {"robot_x": {...}}} style payloads.
				action_items = _expand_robot_keyed_mapping(candidate)
				break
		else:
			if "tasks" in parsed and isinstance(parsed.get("tasks"), Mapping):
				action_items = _expand_robot_keyed_mapping(parsed.get("tasks", {}))
			elif "robot_id" in parsed and ("action" in parsed or "cmd" in parsed or "type" in parsed):
				action_items = [parsed]
			elif ("action" in parsed or "cmd" in parsed or "type" in parsed) and "target" in parsed:
				# Single-robot outputs often omit robot_id.
				action_items = [parsed]
			elif _looks_like_robot_keyed_mapping(parsed):
				action_items = _expand_robot_keyed_mapping(parsed)
			else:
				raise LLMResponseFormatError("JSON object must include list field like 'actions'")
	else:
		raise LLMResponseFormatError("LLM action output must be JSON list/object")

	normalized: List[Dict[str, Any]] = []
	for item in action_items:
		if not isinstance(item, Mapping):
			continue

		robot_id = str(item.get("robot_id", item.get("robot", item.get("car", item.get("ns", item.get("id", "")))))).strip()
		action = str(item.get("action", item.get("cmd", item.get("type", "")))).strip()
		if not action:
			continue

		action_dict: Dict[str, Any] = {"action": action}
		if robot_id:
			action_dict["robot_id"] = robot_id

		if "target" in item:
			action_dict["target"] = item.get("target")
		if "reason" in item and str(item.get("reason", "")).strip():
			action_dict["reason"] = str(item.get("reason")).strip()
		if "mode" in item:
			action_dict["mode"] = item.get("mode")
		if "timeout" in item:
			action_dict["timeout"] = item.get("timeout")
		if "params" in item and isinstance(item.get("params"), Mapping):
			action_dict["params"] = dict(item.get("params", {}))

		normalized.append(action_dict)

	if not normalized:
		raise LLMResponseFormatError("No valid action entries were found")
	return normalized


def _looks_like_robot_keyed_mapping(mapping: Mapping[str, Any]) -> bool:
	if not isinstance(mapping, Mapping):
		return False
	if not mapping:
		return False

	for key, value in mapping.items():
		if not isinstance(key, str):
			return False
		if not isinstance(value, Mapping):
			return False
		if not any(field in value for field in ("action", "cmd", "type")):
			return False
	return True


def _expand_robot_keyed_mapping(mapping: Mapping[str, Any]) -> List[Dict[str, Any]]:
	items: List[Dict[str, Any]] = []
	if not isinstance(mapping, Mapping):
		return items

	for robot_id, payload in mapping.items():
		if not isinstance(payload, Mapping):
			continue
		obj = dict(payload)
		if "robot_id" not in obj and isinstance(robot_id, str):
			obj["robot_id"] = robot_id
		items.append(obj)
	return items


def _extract_message_text(content: Any) -> str:
	if isinstance(content, str):
		return content.strip()

	if isinstance(content, list):
		chunks: List[str] = []
		for part in content:
			if isinstance(part, Mapping) and part.get("type") == "text":
				chunks.append(str(part.get("text", "")))
			elif isinstance(part, str):
				chunks.append(part)
		return "\n".join(chunks).strip()

	if isinstance(content, Mapping):
		return str(content.get("text", "")).strip()

	return ""


def extract_text_from_response(payload: Mapping[str, Any]) -> str:
	"""Extract assistant text from OpenAI-compatible response payload."""
	choices = payload.get("choices")
	if isinstance(choices, list) and choices:
		first_choice = choices[0]
		if isinstance(first_choice, Mapping):
			message = first_choice.get("message")
			if isinstance(message, Mapping):
				text = _extract_message_text(message.get("content"))
				if text:
					return text
			text = str(first_choice.get("text", "")).strip()
			if text:
				return text

	output_text = str(payload.get("output_text", "")).strip()
	if output_text:
		return output_text

	text = str(payload.get("text", "")).strip()
	if text:
		return text

	raise LLMResponseFormatError("Unable to extract text from LLM response payload")


class AsyncLLMClient:
	"""Asynchronous LLM client with retry + concurrency guard."""

	RETRIABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

	def __init__(
		self,
		base_url: str,
		api_key: str = "",
		endpoint: str = "/chat/completions",
		provider: str = "openai_compat",
		max_concurrency: int = 8,
		transport_timeout_s: float = 30.0,
		extra_headers: Optional[Mapping[str, str]] = None,
		log_prompts: bool = False,
		prompt_log_file: str = "",
		prompt_log_console: bool = False,
		split_prompt_logs: bool = False,
		prompt_log_per_run: bool = False,
		prompt_log_run_id: str = "",
		use_openai_sdk: bool = False,
	) -> None:
		base = str(base_url).strip()
		if not base:
			raise ValueError("base_url must not be empty")

		self.base_url = base.rstrip("/")
		self.endpoint = endpoint if endpoint.startswith("/") else "/" + endpoint
		self.provider = str(provider).strip() or "openai_compat"
		self.log_prompts = bool(log_prompts)
		self.prompt_log_file = str(prompt_log_file or "").strip()
		self.prompt_log_console = bool(prompt_log_console)
		self.split_prompt_logs = bool(split_prompt_logs)
		self.prompt_log_per_run = bool(prompt_log_per_run)
		self.prompt_log_run_id = str(prompt_log_run_id or "").strip() or _utc_run_id()
		self._trace_file_lock = asyncio.Lock()
		self.use_openai_sdk = bool(use_openai_sdk)
		if self.use_openai_sdk and self.endpoint != "/chat/completions":
			logger.warning(
				"OpenAI SDK path only supports /chat/completions, got endpoint=%s; fallback to httpx",
				self.endpoint,
			)
			self.use_openai_sdk = False

		concurrency = max(1, int(max_concurrency))
		self._semaphore = asyncio.Semaphore(concurrency)
		self._openai_client = None
		self._client = None

		if self.use_openai_sdk:
			if AsyncOpenAI is None:
				raise ValueError("openai package is required when use_openai_sdk=True")
			resolved_api_key = str(api_key or "").strip() or str(os.getenv("OPENAI_API_KEY", "")).strip()
			if not resolved_api_key:
				logger.warning(
					"OpenAI SDK requested but no API key resolved (llm.api_key/OPENAI_API_KEY empty); fallback to httpx"
				)
				self.use_openai_sdk = False
			else:
				default_headers = dict(extra_headers) if extra_headers else None
				self._openai_client = AsyncOpenAI(
					api_key=resolved_api_key,
					base_url=self.base_url,
					timeout=float(transport_timeout_s),
					default_headers=default_headers,
				)

		if not self.use_openai_sdk:
			headers: Dict[str, str] = {"Content-Type": "application/json"}
			if api_key:
				headers["Authorization"] = "Bearer {}".format(api_key)
			if extra_headers:
				headers.update(dict(extra_headers))

			self._client = httpx.AsyncClient(headers=headers, timeout=float(transport_timeout_s))

	@classmethod
	def from_models_config(cls, models_cfg: Mapping[str, Any]) -> "AsyncLLMClient":
		if not isinstance(models_cfg, Mapping):
			raise ValueError("models_cfg must be a mapping")
		llm_cfg = models_cfg.get("llm", {})
		runtime_cfg = models_cfg.get("runtime", {})
		if not isinstance(llm_cfg, Mapping):
			llm_cfg = {}
		if not isinstance(runtime_cfg, Mapping):
			runtime_cfg = {}

		log_prompts = _as_bool(runtime_cfg.get("log_prompts", False), default=False)
		prompt_log_console = _as_bool(runtime_cfg.get("prompt_log_console", False), default=False)
		prompt_log_file = str(runtime_cfg.get("prompt_log_file", "")).strip()
		split_prompt_logs = _as_bool(runtime_cfg.get("split_prompt_logs", False), default=False)
		prompt_log_per_run = _as_bool(runtime_cfg.get("prompt_log_per_run", False), default=False)
		prompt_log_run_id = str(runtime_cfg.get("prompt_log_run_id", "")).strip()
		use_openai_sdk = _as_bool(llm_cfg.get("use_openai_sdk", False), default=False)

		provider = str(llm_cfg.get("provider", "openai_compat")).strip() or "openai_compat"
		if provider.lower() in ("openai", "openai_sdk"):
			use_openai_sdk = True

		env_log_prompts = os.getenv("MAS_LOG_PROMPTS")
		if env_log_prompts is not None and str(env_log_prompts).strip() != "":
			log_prompts = _as_bool(env_log_prompts, default=log_prompts)

		env_split_prompt_logs = os.getenv("MAS_SPLIT_PROMPT_LOGS")
		if env_split_prompt_logs is not None and str(env_split_prompt_logs).strip() != "":
			split_prompt_logs = _as_bool(env_split_prompt_logs, default=split_prompt_logs)

		env_prompt_log_per_run = os.getenv("MAS_PROMPT_LOG_PER_RUN")
		if env_prompt_log_per_run is not None and str(env_prompt_log_per_run).strip() != "":
			prompt_log_per_run = _as_bool(env_prompt_log_per_run, default=prompt_log_per_run)

		env_prompt_log_run_id = os.getenv("MAS_RUN_ID", "") or os.getenv("MAS_PROMPT_LOG_RUN_ID", "")
		if env_prompt_log_run_id and str(env_prompt_log_run_id).strip() != "":
			prompt_log_run_id = str(env_prompt_log_run_id).strip()

		env_use_openai_sdk = os.getenv("MAS_USE_OPENAI_SDK")
		if env_use_openai_sdk is not None and str(env_use_openai_sdk).strip() != "":
			use_openai_sdk = _as_bool(env_use_openai_sdk, default=use_openai_sdk)

		env_prompt_log_console = os.getenv("MAS_PROMPT_LOG_CONSOLE")
		if env_prompt_log_console is not None and str(env_prompt_log_console).strip() != "":
			prompt_log_console = _as_bool(env_prompt_log_console, default=prompt_log_console)

		env_prompt_log_file = os.getenv("MAS_PROMPT_LOG_FILE")
		if env_prompt_log_file is not None and str(env_prompt_log_file).strip() != "":
			prompt_log_file = str(env_prompt_log_file).strip()

		if log_prompts and not prompt_log_file:
			prompt_log_file = str(Path(__file__).resolve().parent / "logs" / "llm_prompt_trace.log")

		return cls(
			base_url=str(llm_cfg.get("base_url", "")).strip(),
			api_key=str(llm_cfg.get("api_key", "")).strip(),
			endpoint=str(llm_cfg.get("endpoint", "/chat/completions")).strip() or "/chat/completions",
			provider=provider,
			max_concurrency=max(1, _as_int(llm_cfg.get("max_concurrency", 8), "llm.max_concurrency")),
			transport_timeout_s=max(1.0, _as_float(llm_cfg.get("default_timeout_s", 8.0), "llm.default_timeout_s") * 3.0),
			log_prompts=log_prompts,
			prompt_log_file=prompt_log_file,
			prompt_log_console=prompt_log_console,
			split_prompt_logs=split_prompt_logs,
			prompt_log_per_run=prompt_log_per_run,
			prompt_log_run_id=prompt_log_run_id,
			use_openai_sdk=use_openai_sdk,
		)

	async def close(self) -> None:
		if self._client is not None:
			await self._client.aclose()
		if self._openai_client is not None:
			close_func = getattr(self._openai_client, "close", None)
			if callable(close_func):
				result = close_func()
				if asyncio.iscoroutine(result):
					await result

	async def __aenter__(self) -> "AsyncLLMClient":
		return self

	async def __aexit__(self, exc_type, exc, tb) -> None:
		await self.close()

	async def request_text(
		self,
		messages: Sequence[Mapping[str, Any]],
		profile: LLMRequestProfile,
		response_format: Optional[Mapping[str, Any]] = None,
		extra_body: Optional[Mapping[str, Any]] = None,
		trace_tag: str = "",
	) -> str:
		payload = self._build_payload(messages, profile, response_format=response_format, extra_body=extra_body)
		raw = await self._request_json(payload, profile)
		text = extract_text_from_response(raw)

		if self.log_prompts:
			trace_block = self._format_trace_block(
				messages=messages,
				response_text=text,
				trace_tag=trace_tag,
				model=profile.model,
			)
			await self._emit_trace_block(trace_block=trace_block, trace_tag=trace_tag, model=profile.model)

		if self.log_prompts and self.prompt_log_file and (not self.prompt_log_console):
			logger.debug(
				"LLM_OUTPUT tag=%s model=%s preview=%s",
				trace_tag or "-",
				profile.model,
				_single_line_preview(text),
			)
		else:
			logger.info(
				"LLM_OUTPUT tag=%s model=%s preview=%s",
				trace_tag or "-",
				profile.model,
				_single_line_preview(text),
			)
		return text

	def _format_trace_block(
		self,
		messages: Sequence[Mapping[str, Any]],
		response_text: str,
		trace_tag: str,
		model: str,
	) -> str:
		lines: List[str] = []
		lines.append("=" * 96)
		lines.append("LLM_TRACE ts={} tag={} model={}".format(_utc_now_iso(), trace_tag or "-", model))
		lines.append("message_count={}".format(len(messages)))

		for idx, message in enumerate(messages):
			role = str(message.get("role", "")) if isinstance(message, Mapping) else ""
			content = message.get("content", "") if isinstance(message, Mapping) else ""

			if isinstance(content, str):
				content_text = content
			else:
				content_text = json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)

			lines.append("--- message[{}] role={} ---".format(idx, role))
			lines.append(content_text.rstrip())

		lines.append("--- response_text ---")
		lines.append(str(response_text or "").rstrip())
		lines.append("=" * 96)
		lines.append("")
		return "\n".join(lines)

	async def _emit_trace_block(self, trace_block: str, trace_tag: str, model: str) -> None:
		if self.prompt_log_file:
			target_file = self._resolve_trace_file_path(trace_tag)
			await self._append_trace_file(target_file=target_file, trace_block=trace_block)
			logger.debug(
				"LLM_TRACE_WRITTEN tag=%s model=%s file=%s",
				trace_tag or "-",
				model,
				target_file,
			)
			return

		if self.prompt_log_console:
			logger.info("LLM_TRACE tag=%s model=%s\n%s", trace_tag or "-", model, trace_block)
			return

		logger.info("LLM_TRACE tag=%s model=%s (enabled, but no output target configured)", trace_tag or "-", model)

	async def _append_trace_file(self, target_file: str, trace_block: str) -> None:
		if not target_file:
			return

		async with self._trace_file_lock:
			await asyncio.to_thread(self._append_trace_file_sync, target_file, trace_block)

	def _resolve_trace_file_path(self, trace_tag: str) -> str:
		base = Path(self.prompt_log_file).expanduser()
		if not base.is_absolute():
			base = (Path.cwd() / base)

		if (not self.split_prompt_logs) and (not self.prompt_log_per_run):
			return str(base)

		suffix = base.suffix or ".log"
		stem = base.stem if base.suffix else base.name

		name_parts: List[str] = [stem]
		if self.prompt_log_per_run:
			name_parts.append(_sanitize_file_label(self.prompt_log_run_id))
		if self.split_prompt_logs:
			name_parts.append(_trace_bucket(trace_tag))

		filename = "_".join(part for part in name_parts if part) + suffix
		return str(base.parent / filename)

	@staticmethod
	def _append_trace_file_sync(path_text: str, trace_block: str) -> None:
		path = Path(path_text).expanduser()
		if not path.is_absolute():
			path = Path.cwd() / path
		lock = _get_log_file_lock(str(path))
		with lock:
			path.parent.mkdir(parents=True, exist_ok=True)
			with path.open("a", encoding="utf-8") as f:
				f.write(trace_block)

	async def request_actions(
		self,
		messages: Sequence[Mapping[str, Any]],
		profile: LLMRequestProfile,
		response_format: Optional[Mapping[str, Any]] = None,
		extra_body: Optional[Mapping[str, Any]] = None,
		trace_tag: str = "",
	) -> List[Dict[str, Any]]:
		text = await self.request_text(
			messages=messages,
			profile=profile,
			response_format=response_format,
			extra_body=extra_body,
			trace_tag=trace_tag,
		)
		return parse_action_list(text)

	def _build_payload(
		self,
		messages: Sequence[Mapping[str, Any]],
		profile: LLMRequestProfile,
		response_format: Optional[Mapping[str, Any]] = None,
		extra_body: Optional[Mapping[str, Any]] = None,
	) -> Dict[str, Any]:
		payload: Dict[str, Any] = {
			"model": profile.model,
			"messages": list(messages),
			"temperature": float(profile.temperature),
			"max_tokens": int(profile.max_tokens),
			"top_p": float(profile.top_p),
		}
		if response_format:
			payload["response_format"] = dict(response_format)
		if extra_body:
			payload.update(dict(extra_body))
		return payload

	async def _request_json(self, payload: Mapping[str, Any], profile: LLMRequestProfile) -> Dict[str, Any]:
		total_attempts = max(1, int(profile.retries) + 1)
		url = self.base_url + self.endpoint
		last_error: Optional[Exception] = None

		for attempt_idx in range(total_attempts):
			try:
				async with self._semaphore:
					if self.use_openai_sdk:
						data = await self._request_json_via_openai_sdk(payload=payload, profile=profile)
					else:
						data = await self._request_json_via_httpx(url=url, payload=payload, timeout_s=float(profile.timeout_s))

				if not isinstance(data, dict):
					raise LLMAPIError("LLM response payload must be a JSON object")
				self._raise_if_error_payload(data)
				return data

			except _RetriableStatusError as exc:
				last_error = exc
				logger.warning("Retriable LLM HTTP status %s, attempt %s/%s", exc.status_code, attempt_idx + 1, total_attempts)
			except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
				last_error = exc
				logger.warning("Transient LLM transport error on attempt %s/%s: %s", attempt_idx + 1, total_attempts, exc)
			except httpx.HTTPStatusError as exc:
				status = exc.response.status_code if exc.response is not None else "unknown"
				body = exc.response.text[:500] if exc.response is not None else str(exc)
				raise LLMAPIError("Non-retriable LLM HTTP error {}: {}".format(status, body))
			except Exception as exc:
				if self.use_openai_sdk and self._looks_like_openai_error(exc):
					status_code = self._extract_openai_status_code(exc)
					if status_code in self.RETRIABLE_HTTP_STATUS or status_code <= 0:
						last_error = exc
						logger.warning("Transient OpenAI SDK error on attempt %s/%s: %s", attempt_idx + 1, total_attempts, exc)
					else:
						raise LLMAPIError("Non-retriable LLM HTTP error {}: {}".format(status_code, exc))
				else:
					raise
			except ValueError as exc:
				raise LLMAPIError("LLM response was not valid JSON: {}".format(exc))

			if attempt_idx >= total_attempts - 1:
				break

			base_backoff = max(0.05, float(profile.backoff_s))
			delay_s = base_backoff * (2 ** attempt_idx) + random.uniform(0.0, base_backoff * 0.3)
			await asyncio.sleep(delay_s)

		raise LLMAPIError("LLM request failed after {} attempts: {}".format(total_attempts, last_error))

	async def _request_json_via_httpx(self, url: str, payload: Mapping[str, Any], timeout_s: float) -> Dict[str, Any]:
		if self._client is None:
			raise LLMAPIError("HTTP transport client is not initialized")

		response = await self._client.post(
			url,
			json=dict(payload),
			timeout=float(timeout_s),
		)

		if response.status_code in self.RETRIABLE_HTTP_STATUS:
			raise _RetriableStatusError(response.status_code, response.text[:400])

		response.raise_for_status()
		data = response.json()
		if not isinstance(data, dict):
			raise LLMAPIError("LLM response payload must be a JSON object")
		return data

	async def _request_json_via_openai_sdk(
		self,
		payload: Mapping[str, Any],
		profile: LLMRequestProfile,
	) -> Dict[str, Any]:
		if self._openai_client is None:
			raise LLMAPIError("OpenAI SDK client is not initialized")

		request_payload = dict(payload)
		request_payload["timeout"] = float(profile.timeout_s)
		response = await self._openai_client.chat.completions.create(**request_payload)

		if hasattr(response, "model_dump"):
			data = response.model_dump()
		elif hasattr(response, "to_dict"):
			data = response.to_dict()
		else:
			raise LLMAPIError("OpenAI SDK response object does not support dict conversion")

		if not isinstance(data, dict):
			raise LLMAPIError("OpenAI SDK response payload must be a JSON object")

		return data

	@staticmethod
	def _raise_if_error_payload(data: Mapping[str, Any]) -> None:
		error_obj = data.get("error")
		if isinstance(error_obj, Mapping):
			error_msg = str(error_obj.get("message", error_obj)).strip()
			error_code = str(error_obj.get("code", "")).strip()
			raise LLMAPIError(
				"LLM error payload{}: {}".format(
					(" code=" + error_code) if error_code else "",
					error_msg or "unknown error",
				)
			)

	@staticmethod
	def _looks_like_openai_error(exc: Exception) -> bool:
		module_name = str(exc.__class__.__module__)
		return module_name.startswith("openai")

	@staticmethod
	def _extract_openai_status_code(exc: Exception) -> int:
		status = getattr(exc, "status_code", None)
		if status is None:
			response = getattr(exc, "response", None)
			status = getattr(response, "status_code", None)
		try:
			return int(status)
		except (TypeError, ValueError):
			return -1


__all__ = [
	"AsyncLLMClient",
	"LLMAPIError",
	"LLMRequestProfile",
	"LLMResponseFormatError",
	"build_messages",
	"build_profile_from_models",
	"extract_text_from_response",
	"parse_action_list",
	"render_prompt",
]


_LOG_FILE_LOCKS: Dict[str, threading.Lock] = {}
_LOG_FILE_LOCKS_GUARD = threading.Lock()


def _get_log_file_lock(path_text: str) -> threading.Lock:
	key = str(path_text)
	with _LOG_FILE_LOCKS_GUARD:
		lock = _LOG_FILE_LOCKS.get(key)
		if lock is None:
			lock = threading.Lock()
			_LOG_FILE_LOCKS[key] = lock
	return lock


def _trace_bucket(trace_tag: str) -> str:
	tag = str(trace_tag or "").strip().lower()
	if tag.startswith("leader:"):
		return "leader"
	if tag.startswith("car:"):
		return "car"
	return "other"


def _sanitize_file_label(value: str) -> str:
	text = str(value or "").strip()
	if not text:
		return "run"
	chars: List[str] = []
	for ch in text:
		if ch.isalnum() or ch in ("-", "_"):
			chars.append(ch)
		else:
			chars.append("_")
	cleaned = "".join(chars).strip("_")
	return cleaned or "run"


def _utc_run_id() -> str:
	return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
