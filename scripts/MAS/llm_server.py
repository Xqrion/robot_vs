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
from typing import Any, Dict, Mapping

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from config_loader import ConfigError, ConfigLoader
from mas_manager import HierarchicalMASManager


LOGGER = logging.getLogger("mas.llm_server")


def _to_int(value: Any, default: int) -> int:
	try:
		return int(value)
	except (TypeError, ValueError):
		return int(default)


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
