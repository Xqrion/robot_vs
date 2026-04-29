#!/usr/bin/env python3
"""Probe MAS LLM response structure and text extraction compatibility.

Usage example:
  python scripts/MAS/llm_response_probe.py 
    --section leader_model 
    --prompt-role leader 
    --model kimi-k2.5 
    --side red 
    --print-extracted
	
	or for no WIFI api call dry-run:
	
	python llm_response_probe.py --section leader_model --prompt-role leader --model kimi-k2.5 --side red --dry-run

The script will:
1) load MAS models/prompts config,
2) issue one real request to the configured endpoint,
3) save raw response JSON,
4) run extract_text_from_response on the raw payload,
5) write a readable report for debugging extraction mismatch.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import replace
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
	sys.path.insert(0, str(_THIS_DIR))

try:
	from config_loader import ConfigError, ConfigLoader
	from llm_api import (
		AsyncLLMClient,
		LLMAPIError,
		LLMRequestProfile,
		LLMResponseFormatError,
		build_messages,
		build_profile_from_models,
		extract_text_from_response,
		render_prompt,
	)
except ImportError:  # pragma: no cover
	from .config_loader import ConfigError, ConfigLoader  # type: ignore
	from .llm_api import (  # type: ignore
		AsyncLLMClient,
		LLMAPIError,
		LLMRequestProfile,
		LLMResponseFormatError,
		build_messages,
		build_profile_from_models,
		extract_text_from_response,
		render_prompt,
	)


LOGGER = logging.getLogger("mas.llm_response_probe")


def _utc_stamp() -> str:
	return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _safe_preview(value: Any, max_chars: int = 180) -> str:
	text = " ".join(str(value or "").split())
	if len(text) <= max_chars:
		return text
	if max_chars <= 3:
		return text[:max_chars]
	return text[: max_chars - 3] + "..."


def _safe_mapping(value: Any) -> Dict[str, Any]:
	if isinstance(value, Mapping):
		return dict(value)
	return {}


def _sanitize_filename(text: str) -> str:
	raw = str(text or "").strip()
	if not raw:
		return "unknown"
	chars: List[str] = []
	for ch in raw:
		if ch.isalnum() or ch in ("-", "_"):
			chars.append(ch)
		else:
			chars.append("_")
	out = "".join(chars).strip("_")
	return out or "unknown"


def _resolve_api_key_for_side(side: str, default_api_key: Any) -> Tuple[str, str]:
	normalized_side = str(side or "").strip().lower()
	side_key_name = "LLM_API_KEY_{}".format(normalized_side.upper()) if normalized_side else ""
	if side_key_name:
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


def _prepare_models_cfg(base_models: Mapping[str, Any], side: str) -> Tuple[Dict[str, Any], str]:
	models_cfg = copy.deepcopy(dict(base_models))
	llm_cfg = _safe_mapping(models_cfg.get("llm", {}))

	api_key, source = _resolve_api_key_for_side(side=side, default_api_key=llm_cfg.get("api_key", ""))
	llm_cfg["api_key"] = api_key
	llm_cfg["api_key_source"] = source
	models_cfg["llm"] = llm_cfg
	return models_cfg, source


def _default_leader_prompt_inputs(side: str) -> Dict[str, Any]:
	normalized_side = str(side or "red").strip().lower() or "red"
	my_robot_a = "robot_{}_1".format(normalized_side)
	my_robot_b = "robot_{}_2".format(normalized_side)

	global_state = {
		"team_color": normalized_side,
		"my_cars": [my_robot_a, my_robot_b],
		"friendly": {
			my_robot_a: {"state": {"alive": True, "hp": 92, "ammo": 30, "in_combat": False}, "stale": False},
			my_robot_b: {"state": {"alive": True, "hp": 74, "ammo": 16, "in_combat": True}, "stale": False},
		},
		"enemy": {
			"stale": False,
			"state": {
				"visible_count": 1,
				"enemies": [{"id": "enemy_1", "hp": 46, "x": 1.2, "y": -0.4}],
			},
		},
	}

	return {
		"global_state": global_state,
		"stm_summary": "Last 2 cycles: one ally took damage; enemy stayed near center lane.",
		"ltm_summary": "Historically, split-angle pressure works better than direct rush.",
	}


def _default_car_prompt_inputs(side: str, robot_id: str) -> Dict[str, Any]:
	normalized_side = str(side or "red").strip().lower() or "red"
	normalized_robot = str(robot_id or "robot_{}_1".format(normalized_side)).strip() or "robot_{}_1".format(normalized_side)

	car_state = {
		"side": normalized_side,
		"robot_id": normalized_robot,
		"local_state": {
			"robot_id": normalized_robot,
			"team_color": normalized_side,
			"alive": True,
			"hp": 88.0,
			"ammo": 20.0,
			"in_combat": True,
			"yaw": 0.2,
			"visible_enemies": [{"id": "enemy_1", "x": 1.0, "y": -0.5}],
			"safe_point": {"x": 0.0, "y": 0.0},
		},
	}

	team_context = {
		"team_color": normalized_side,
		"my_cars": [normalized_robot],
		"friendly": {
			normalized_robot: {
				"stale": False,
				"state": {
					"alive": True,
					"hp": 88.0,
					"ammo": 20.0,
					"in_combat": True,
				},
			}
		},
		"enemy": {
			"stale": False,
			"state": {
				"visible_count": 1,
				"enemies": [{"id": "enemy_1", "x": 1.0, "y": -0.5, "hp": 40}],
			},
		},
	}

	return {
		"leader_order": "Keep pressure but do not over-extend.",
		"car_state": car_state,
		"team_context": team_context,
	}


def _build_probe_messages(
	prompts_cfg: Mapping[str, Any],
	prompt_role: str,
	side: str,
	robot_id: str,
) -> Sequence[Dict[str, str]]:
	role = str(prompt_role or "leader").strip().lower()
	role_cfg = _safe_mapping(prompts_cfg.get(role, {}))

	system_prompt = str(role_cfg.get("system_prompt", "")).strip()
	user_template = str(role_cfg.get("user_template", "")).strip()

	if not system_prompt or not user_template:
		raise ValueError("Prompt role {} is missing system_prompt/user_template".format(role))

	if role == "leader":
		inputs = _default_leader_prompt_inputs(side=side)
	elif role == "car":
		inputs = _default_car_prompt_inputs(side=side, robot_id=robot_id)
	else:
		raise ValueError("Unsupported prompt_role: {} (expected leader or car)".format(prompt_role))

	user_prompt = render_prompt(user_template, **inputs)
	return build_messages(system_prompt=system_prompt, user_prompt=user_prompt)


def _collect_string_fields(value: Any, path: str = "$") -> List[Tuple[str, str]]:
	result: List[Tuple[str, str]] = []

	if isinstance(value, str):
		trimmed = value.strip()
		if trimmed:
			result.append((path, trimmed))
		return result

	if isinstance(value, Mapping):
		for key, item in value.items():
			child_path = "{}.{}".format(path, key)
			result.extend(_collect_string_fields(item, path=child_path))
		return result

	if isinstance(value, list):
		for idx, item in enumerate(value):
			child_path = "{}[{}]".format(path, idx)
			result.extend(_collect_string_fields(item, path=child_path))

	return result


def _candidate_text_fields(payload: Mapping[str, Any], max_items: int) -> List[Tuple[str, str]]:
	all_fields = _collect_string_fields(payload)
	keywords = ("text", "content", "message", "output", "reason", "response", "answer")

	candidates: List[Tuple[str, str]] = []
	for path, value in all_fields:
		path_lower = path.lower()
		if any(key in path_lower for key in keywords):
			candidates.append((path, value))
			if len(candidates) >= max_items:
				return candidates

	# Fallback: return first non-empty strings when keyword matching gives nothing.
	for path, value in all_fields:
		if len(candidates) >= max_items:
			break
		candidates.append((path, value))

	return candidates


def _dump_json(path: Path, payload: Any) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _dump_text(path: Path, text: str) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		f.write(text)


def _profile_with_overrides(profile: LLMRequestProfile, args: argparse.Namespace) -> LLMRequestProfile:
	updated = profile
	if args.model:
		updated = replace(updated, model=str(args.model).strip())
	if args.temperature is not None:
		updated = replace(updated, temperature=float(args.temperature))
	if args.max_tokens is not None:
		updated = replace(updated, max_tokens=int(args.max_tokens))
	if args.top_p is not None:
		updated = replace(updated, top_p=float(args.top_p))
	if args.timeout_s is not None:
		updated = replace(updated, timeout_s=float(args.timeout_s))
	if args.retries is not None:
		updated = replace(updated, retries=int(args.retries))
	if args.backoff_s is not None:
		updated = replace(updated, backoff_s=float(args.backoff_s))
	return updated


def _parse_optional_json(raw: str, field_name: str) -> Optional[Mapping[str, Any]]:
	text = str(raw or "").strip()
	if not text:
		return None
	try:
		parsed = json.loads(text)
	except Exception as exc:
		raise ValueError("{} must be valid JSON object: {}".format(field_name, exc))
	if not isinstance(parsed, Mapping):
		raise ValueError("{} must decode to JSON object".format(field_name))
	return dict(parsed)


async def _async_main(args: argparse.Namespace) -> int:
	configs_root = Path(args.configs_root).expanduser().resolve()
	loader = ConfigLoader(root_dir=configs_root)
	bundle = loader.load_all()

	models_cfg, api_key_source = _prepare_models_cfg(bundle.models, side=args.side)
	profile = build_profile_from_models(models_cfg=models_cfg, section_name=args.section)
	profile = _profile_with_overrides(profile=profile, args=args)

	messages = _build_probe_messages(
		prompts_cfg=bundle.prompts,
		prompt_role=args.prompt_role,
		side=args.side,
		robot_id=args.robot_id,
	)

	response_format = _parse_optional_json(args.response_format, "--response-format")
	extra_body = _parse_optional_json(args.extra_body, "--extra-body")

	client = AsyncLLMClient.from_models_config(models_cfg)
	raw_payload: Dict[str, Any] = {}
	request_payload: Dict[str, Any] = {}

	try:
		# This probe intentionally inspects raw payload shape, so it uses internal request helpers.
		request_payload = client._build_payload(
			messages=messages,
			profile=profile,
			response_format=response_format,
			extra_body=extra_body,
		)

		if args.dry_run:
			raw_payload = {
				"dry_run": True,
				"note": "No outbound request made.",
				"request_payload": request_payload,
			}
		else:
			raw_payload = await client._request_json(payload=request_payload, profile=profile)
	finally:
		await client.close()

	extracted_text = ""
	extraction_error = ""
	extraction_ok = False

	if not args.dry_run:
		try:
			extracted_text = extract_text_from_response(raw_payload)
			extraction_ok = True
		except LLMResponseFormatError as exc:
			extraction_error = str(exc)
		except Exception as exc:
			extraction_error = "Unexpected extraction error: {}".format(exc)

	top_keys = sorted(list(raw_payload.keys())) if isinstance(raw_payload, Mapping) else []
	field_candidates = _candidate_text_fields(raw_payload, max_items=max(1, int(args.max_candidates)))

	output_dir = Path(args.output_dir).expanduser().resolve()
	stamp = _utc_stamp()
	model_label = _sanitize_filename(profile.model)
	prefix = "llm_probe_{}_{}_{}_{}".format(args.prompt_role, args.section, model_label, stamp)

	raw_path = output_dir / (prefix + "_raw.json")
	report_path = output_dir / (prefix + "_report.txt")
	extracted_path = output_dir / (prefix + "_extracted.txt")

	_dump_json(raw_path, raw_payload)

	report_lines: List[str] = []
	report_lines.append("LLM RESPONSE PROBE REPORT")
	report_lines.append("timestamp_utc={}".format(datetime.now(timezone.utc).isoformat(timespec="seconds")))
	report_lines.append("configs_root={}".format(configs_root))
	report_lines.append("section={}".format(args.section))
	report_lines.append("prompt_role={}".format(args.prompt_role))
	report_lines.append("side={}".format(args.side))
	report_lines.append("robot_id={}".format(args.robot_id))
	report_lines.append("model={}".format(profile.model))
	report_lines.append("api_key_source={}".format(api_key_source))
	report_lines.append("dry_run={}".format(bool(args.dry_run)))
	report_lines.append("request_timeout_s={}".format(profile.timeout_s))
	report_lines.append("request_retries={}".format(profile.retries))
	report_lines.append("top_level_keys={}".format(top_keys))
	report_lines.append("extract_ok={}".format(extraction_ok))
	if extraction_error:
		report_lines.append("extract_error={}".format(extraction_error))
	report_lines.append("")
	report_lines.append("request_payload_preview={}".format(_safe_preview(request_payload)))
	report_lines.append("")
	report_lines.append("candidate_text_fields:")
	for idx, (path, value) in enumerate(field_candidates, start=1):
		report_lines.append("{}. {} => {}".format(idx, path, _safe_preview(value, max_chars=240)))

	if extraction_ok:
		report_lines.append("")
		report_lines.append("extracted_text_preview={}".format(_safe_preview(extracted_text, max_chars=400)))

	_dump_text(report_path, "\n".join(report_lines) + "\n")

	if extraction_ok:
		_dump_text(extracted_path, extracted_text + "\n")

	print("[probe] raw response saved: {}".format(raw_path))
	print("[probe] report saved: {}".format(report_path))
	if extraction_ok:
		print("[probe] extraction: OK")
		print("[probe] extracted text saved: {}".format(extracted_path))
		if args.print_extracted:
			print("\n----- extracted_text -----")
			print(extracted_text)
			print("----- extracted_text end -----")
	else:
		if args.dry_run:
			print("[probe] extraction skipped because --dry-run is enabled")
		else:
			print("[probe] extraction: FAILED ({})".format(extraction_error or "unknown error"))

	if args.strict and (not args.dry_run) and (not extraction_ok):
		return 2
	return 0


def _build_arg_parser() -> argparse.ArgumentParser:
	this_dir = Path(__file__).resolve().parent
	default_output = (this_dir.parents[1] / "debug").resolve()

	parser = argparse.ArgumentParser(description="Probe MAS LLM response structure for extraction debugging")
	parser.add_argument("--configs-root", type=str, default=str(this_dir), help="MAS root dir containing configs/")
	parser.add_argument(
		"--section",
		type=str,
		default="leader_model",
		choices=("leader_model", "car_model"),
		help="Model profile section from models config",
	)
	parser.add_argument(
		"--prompt-role",
		type=str,
		default="leader",
		choices=("leader", "car"),
		help="Prompt template role to render",
	)
	parser.add_argument("--model", type=str, default="", help="Optional model name override")
	parser.add_argument("--side", type=str, default="red", choices=("red", "blue"), help="Team side for key resolution")
	parser.add_argument("--robot-id", type=str, default="robot_red_1", help="Robot id for car prompt role")

	parser.add_argument("--temperature", type=float, default=None, help="Optional override for temperature")
	parser.add_argument("--max-tokens", type=int, default=None, help="Optional override for max_tokens")
	parser.add_argument("--top-p", type=float, default=None, help="Optional override for top_p")
	parser.add_argument("--timeout-s", type=float, default=None, help="Optional override for timeout_s")
	parser.add_argument("--retries", type=int, default=None, help="Optional override for retries")
	parser.add_argument("--backoff-s", type=float, default=None, help="Optional override for backoff_s")

	parser.add_argument("--response-format", type=str, default="", help="Optional JSON object string")
	parser.add_argument("--extra-body", type=str, default="", help="Optional JSON object string")
	parser.add_argument("--max-candidates", type=int, default=20, help="Max candidate string fields in report")

	parser.add_argument("--output-dir", type=str, default=str(default_output), help="Directory for raw/report output")
	parser.add_argument("--print-extracted", action="store_true", help="Print extracted text to console")
	parser.add_argument("--dry-run", action="store_true", help="Build payload only, skip outbound request")
	parser.add_argument("--strict", action="store_true", help="Exit non-zero when extraction fails")
	parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

	return parser


def main() -> None:
	parser = _build_arg_parser()
	args = parser.parse_args()

	logging.basicConfig(
		level=logging.DEBUG if args.verbose else logging.INFO,
		format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
	)

	try:
		exit_code = asyncio.run(_async_main(args))
	except (ConfigError, ValueError, LLMAPIError, LLMResponseFormatError) as exc:
		LOGGER.error("Probe failed: %s", exc)
		raise SystemExit(1)
	except KeyboardInterrupt:
		LOGGER.info("Interrupted by user")
		raise SystemExit(130)

	raise SystemExit(exit_code)


if __name__ == "__main__":
	main()
