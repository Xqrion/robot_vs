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
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import httpx


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
	) -> None:
		base = str(base_url).strip()
		if not base:
			raise ValueError("base_url must not be empty")

		self.base_url = base.rstrip("/")
		self.endpoint = endpoint if endpoint.startswith("/") else "/" + endpoint
		self.provider = str(provider).strip() or "openai_compat"

		concurrency = max(1, int(max_concurrency))
		self._semaphore = asyncio.Semaphore(concurrency)

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
		if not isinstance(llm_cfg, Mapping):
			llm_cfg = {}

		return cls(
			base_url=str(llm_cfg.get("base_url", "")).strip(),
			api_key=str(llm_cfg.get("api_key", "")).strip(),
			endpoint=str(llm_cfg.get("endpoint", "/chat/completions")).strip() or "/chat/completions",
			provider=str(llm_cfg.get("provider", "openai_compat")).strip() or "openai_compat",
			max_concurrency=max(1, _as_int(llm_cfg.get("max_concurrency", 8), "llm.max_concurrency")),
			transport_timeout_s=max(1.0, _as_float(llm_cfg.get("default_timeout_s", 8.0), "llm.default_timeout_s") * 3.0),
		)

	async def close(self) -> None:
		await self._client.aclose()

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
		logger.info("LLM_OUTPUT tag=%s model=%s text=%s", trace_tag or "-", profile.model, text)
		return text

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
					response = await self._client.post(
						url,
						json=dict(payload),
						timeout=float(profile.timeout_s),
					)

				if response.status_code in self.RETRIABLE_HTTP_STATUS:
					raise _RetriableStatusError(response.status_code, response.text[:400])

				response.raise_for_status()
				data = response.json()
				if not isinstance(data, dict):
					raise LLMAPIError("LLM response payload must be a JSON object")
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
			except ValueError as exc:
				raise LLMAPIError("LLM response was not valid JSON: {}".format(exc))

			if attempt_idx >= total_attempts - 1:
				break

			base_backoff = max(0.05, float(profile.backoff_s))
			delay_s = base_backoff * (2 ** attempt_idx) + random.uniform(0.0, base_backoff * 0.3)
			await asyncio.sleep(delay_s)

		raise LLMAPIError("LLM request failed after {} attempts: {}".format(total_attempts, last_error))


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
