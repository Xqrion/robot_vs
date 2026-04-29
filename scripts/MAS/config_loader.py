#!/usr/bin/env python3
"""Configuration loader for the hierarchical MAS stack.

This module provides a small but robust layer on top of YAML configs:
1) default-value injection,
2) environment-variable override,
3) shape validation,
4) light file caching based on mtime.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple

import yaml


class ConfigError(RuntimeError):
	"""Raised when configuration cannot be loaded or validated."""


def _ensure_dict(value: Any, source: str) -> Dict[str, Any]:
	if value is None:
		return {}
	if not isinstance(value, dict):
		raise ConfigError("Config at {} must be a mapping/dict".format(source))
	return value


def _read_yaml_file(path: Path) -> Dict[str, Any]:
	try:
		with path.open("r", encoding="utf-8") as f:
			data = yaml.safe_load(f)
	except FileNotFoundError:
		raise ConfigError("Config file not found: {}".format(path))
	except yaml.YAMLError as exc:
		raise ConfigError("Invalid YAML in {}: {}".format(path, exc))
	except OSError as exc:
		raise ConfigError("Failed to read {}: {}".format(path, exc))

	return _ensure_dict(data, str(path))


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
	merged = copy.deepcopy(dict(base))
	for key, value in override.items():
		if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
			merged[key] = _deep_merge(merged[key], value)
		else:
			merged[key] = copy.deepcopy(value)
	return merged


def _set_nested(mapping: MutableMapping[str, Any], keys: Tuple[str, ...], value: Any) -> None:
	cursor = mapping
	for key in keys[:-1]:
		next_level = cursor.get(key)
		if not isinstance(next_level, dict):
			next_level = {}
			cursor[key] = next_level
		cursor = next_level
	cursor[keys[-1]] = value


def _require_positive_number(value: Any, field_name: str) -> float:
	try:
		parsed = float(value)
	except (TypeError, ValueError):
		raise ConfigError("{} must be a number, got {}".format(field_name, value))
	if parsed <= 0:
		raise ConfigError("{} must be > 0, got {}".format(field_name, value))
	return parsed


@dataclass(frozen=True)
class MASConfigBundle:
	models: Dict[str, Any]
	prompts: Dict[str, Any]

### zhe ge lei bu zhi dao gan ma de !! change change
class ConfigLoader:
	"""Load MAS configs from local configs folder with legacy fallback support."""

	DEFAULT_MODELS: Dict[str, Any] = {
		"llm": {
			"provider": "openai_compat",
			"base_url": "http://127.0.0.1:8000/v1",
			"api_key": "",
			"endpoint": "/chat/completions",
			"default_timeout_s": 8.0,
			"default_retries": 2,
			"default_backoff_s": 0.4,
			"max_concurrency": 8,
		},
		"leader_model": {
			"name": "gpt-4o-mini",
			"temperature": 0.3,
			"max_tokens": 512,
			"top_p": 0.95,
			"timeout_s": 10.0,
			"retries": 2,
			"backoff_s": 0.5,
		},
		"car_model": {
			"name": "gpt-4o-mini",
			"temperature": 0.2,
			"max_tokens": 256,
			"top_p": 0.9,
			"timeout_s": 4.0,
			"retries": 1,
			"backoff_s": 0.3,
		},
		"runtime": {
			"leader_loop_interval_s": 5.0,
			"car_loop_interval_s": 1.0,
			"team_ports": {"red": 8001, "blue": 8002},
		},
	}

	MODEL_ENV_OVERRIDES: Dict[str, Tuple[Tuple[str, ...], Callable[[str], Any]]] = {
		"SITP_LLM_PROVIDER": (("llm", "provider"), str),
		"SITP_LLM_BASE_URL": (("llm", "base_url"), str),
		"SITP_LLM_API_KEY": (("llm", "api_key"), str),
		"SITP_LLM_ENDPOINT": (("llm", "endpoint"), str),
		"SITP_LLM_DEFAULT_TIMEOUT_S": (("llm", "default_timeout_s"), float),
		"SITP_LLM_DEFAULT_RETRIES": (("llm", "default_retries"), int),
		"SITP_LLM_MAX_CONCURRENCY": (("llm", "max_concurrency"), int),
		"SITP_LEADER_MODEL": (("leader_model", "name"), str),
		"SITP_CAR_MODEL": (("car_model", "name"), str),
		"SITP_LEADER_LOOP_INTERVAL_S": (("runtime", "leader_loop_interval_s"), float),
		"SITP_CAR_LOOP_INTERVAL_S": (("runtime", "car_loop_interval_s"), float),
	}

	def __init__(self, root_dir: Optional[Path] = None, configs_dir_name: str = "configs") -> None:
		package_root = Path(__file__).resolve().parent
		self.root_dir = Path(root_dir).resolve() if root_dir is not None else package_root
		self.configs_dir = self.root_dir / configs_dir_name
		self.legacy_config_dir = self.root_dir.parents[1] / "config" / "MAS"
		self.prompts_file_name = str(os.getenv("MAS_PROMPTS_FILE", "")).strip()
		self.prompts_path_raw = str(os.getenv("MAS_PROMPTS_PATH", "")).strip()
		self._cache: Dict[Path, Tuple[int, Dict[str, Any]]] = {}

	def _prompt_candidates(self) -> Tuple[Path, ...]:
		candidates: List[Path] = []

		if self.prompts_path_raw:
			prompt_path = Path(self.prompts_path_raw).expanduser()
			if not prompt_path.is_absolute():
				prompt_path = (self.root_dir / prompt_path).resolve()
			candidates.append(prompt_path)

		if self.prompts_file_name:
			candidates.append(self.configs_dir / self.prompts_file_name)
			candidates.append(self.legacy_config_dir / self.prompts_file_name)

		# Default search order with legacy fallback compatibility.
		candidates.extend(
			[
				self.configs_dir / "prompts.yaml",
				self.legacy_config_dir / "prompts.yaml",
				self.legacy_config_dir / "prompt.yaml",
			]
		)

		ordered_unique: List[Path] = []
		seen = set()
		for item in candidates:
			key = str(item)
			if key in seen:
				continue
			seen.add(key)
			ordered_unique.append(item)

		return tuple(ordered_unique)

	def _load_yaml_with_cache(self, path: Path) -> Dict[str, Any]:
		stat = path.stat()
		stamp = stat.st_mtime_ns
		cached = self._cache.get(path)
		if cached and cached[0] == stamp:
			return copy.deepcopy(cached[1])

		loaded = _read_yaml_file(path)
		self._cache[path] = (stamp, loaded)
		return copy.deepcopy(loaded)

	@staticmethod
	def _first_existing(paths: Tuple[Path, ...], label: str) -> Path:
		for path in paths:
			if path.exists() and path.is_file():
				return path
		path_list = ", ".join(str(p) for p in paths)
		raise ConfigError("No {} config found. Tried: {}".format(label, path_list))

	def _apply_model_defaults(self, models: Dict[str, Any]) -> Dict[str, Any]:
		return _deep_merge(self.DEFAULT_MODELS, models)

	def _apply_env_overrides(self, models: Dict[str, Any]) -> Dict[str, Any]:
		merged = copy.deepcopy(models)
		for env_key, (target_path, caster) in self.MODEL_ENV_OVERRIDES.items():
			raw = os.getenv(env_key)
			if raw is None or raw == "":
				continue
			try:
				value = caster(raw)
			except (TypeError, ValueError):
				raise ConfigError("Invalid env override {}={}, cast failed".format(env_key, raw))
			_set_nested(merged, target_path, value)
		return merged

	@staticmethod
	def _validate_models(models: Dict[str, Any]) -> None:
		for section in ("llm", "leader_model", "car_model", "runtime"):
			if not isinstance(models.get(section), dict):
				raise ConfigError("models.{} must be a dict".format(section))

		for model_key in ("leader_model", "car_model"):
			model_name = str(models[model_key].get("name", "")).strip()
			if not model_name:
				raise ConfigError("models.{}.name must not be empty".format(model_key))

		_require_positive_number(models["llm"].get("default_timeout_s"), "models.llm.default_timeout_s")
		_require_positive_number(models["llm"].get("max_concurrency"), "models.llm.max_concurrency")
		_require_positive_number(models["runtime"].get("leader_loop_interval_s"), "models.runtime.leader_loop_interval_s")
		_require_positive_number(models["runtime"].get("car_loop_interval_s"), "models.runtime.car_loop_interval_s")

	@staticmethod
	def _validate_prompts(prompts: Dict[str, Any]) -> None:
		for role in ("leader", "car"):
			role_cfg = prompts.get(role)
			if not isinstance(role_cfg, dict):
				raise ConfigError("prompts.{} must be a dict".format(role))
			for required_key in ("system_prompt", "user_template"):
				text = str(role_cfg.get(required_key, "")).strip()
				if not text:
					raise ConfigError("prompts.{}.{} must not be empty".format(role, required_key))

	def load_models(self) -> Dict[str, Any]:
		candidate = self._first_existing(
			(
				self.configs_dir / "models.yaml",
				self.legacy_config_dir / "models.yaml",
			),
			label="models",
		)
		loaded = self._load_yaml_with_cache(candidate)
		merged = self._apply_model_defaults(loaded)
		merged = self._apply_env_overrides(merged)
		self._validate_models(merged)
		return merged

	def load_prompts(self) -> Dict[str, Any]:
		candidate = self._first_existing(
			self._prompt_candidates(),
			label="prompts",
		)
		loaded = self._load_yaml_with_cache(candidate)
		self._validate_prompts(loaded)
		return loaded

	def load_all(self) -> MASConfigBundle:
		return MASConfigBundle(models=self.load_models(), prompts=self.load_prompts())

	def reload(self) -> MASConfigBundle:
		self._cache.clear()
		return self.load_all()


def get_config_loader(root_dir: Optional[Path] = None) -> ConfigLoader:
	return ConfigLoader(root_dir=root_dir)


def load_all_configs(root_dir: Optional[Path] = None) -> MASConfigBundle:
	return get_config_loader(root_dir=root_dir).load_all()


__all__ = [
	"ConfigError",
	"ConfigLoader",
	"MASConfigBundle",
	"get_config_loader",
	"load_all_configs",
]
