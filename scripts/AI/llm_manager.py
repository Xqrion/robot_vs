### 在miniconda的robotvs下运行

import argparse
import json
import os
import time

import yaml
from fastapi import Body, FastAPI, HTTPException
from openai import OpenAI
import uvicorn


class LLMManager(object):
    """LLM API 适配器：负责构造提示词并返回可执行任务 JSON。
        Config优先级是 config.yaml → env → LLMManager默认值
    """

    def __init__(self, api_key=None, base_url=None, model="", timeout_s=30.0):
        self.api_key = str(api_key or self._read_api_key())
        self.base_url = str(base_url or self._read_base_url())
        self.model = str(model)
        self.timeout_s = float(timeout_s)

        if not self.api_key:
            raise ValueError("LLM API key is empty. Set env LLM_API_KEY or LLM_API")
        if not self.base_url:
            raise ValueError("LLM base_url is empty. Configure it in llm_config.yaml")
        if not self.model:
            raise ValueError("LLM model_name is empty. Configure it in llm_config.yaml")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _read_api_key(self):
        return os.getenv("LLM_API_KEY", "") or os.getenv("LLM_API", "")

    def _read_base_url(self):
        return os.getenv("LLM_BASE_URL", "")

    def ask_raw(self, prompt):
        """向 LLM 发送纯文本 prompt，返回模型原始文本。"""
        # print("[llm_manager] sending prompt to LLM:")
        # print(str(prompt))
        request_start_s = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是多机器人战术规划器。"
                        "你必须只输出一个 JSON 对象，不能输出解释、markdown 或额外文本。"
                    ),
                },
                {"role": "user", "content": str(prompt)},
            ],
            timeout=self.timeout_s,

            # for Qwen3.5-flash
            extra_body={
                "enable_thinking": False
            },
        )
        request_end_s = time.time()
        request_elapsed_s = request_end_s - request_start_s
        request_time_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(request_end_s))
        raw_text = str(response.choices[0].message.content or "").strip()
        # print("[llm_manager] raw LLM response:")
        # print(raw_text)
        print("[llm_manager] request_time={}, elapsed_s={:.3f}".format(
            request_time_text,
            request_elapsed_s,
        ))
        return raw_text

    def build_prompt(self, battle_state, robot_ids):
        payload = {
            "battle_state": battle_state,
            "robot_ids": robot_ids,
            "required_output": {
                "<robot_id>": {
                    "action": "STOP | GOTO | ATTACK| ROTATE",
                    "target": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                    "mode": "int",
                    "reason": "short string",
                    "timeout": "float",
                }
            },
            "constraints": [
                "只输出 JSON 对象",
                "必须覆盖 robot_ids 里每一台车",
                "target 必须包含 x/y 数值",
                "timeout 必须 > 5",
                "地图x范围[-3.8, 3.8], y范围[-1.8, 1.8]",
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    def plan_tasks(self, battle_state, robot_ids):
        prompt = self.build_prompt(battle_state=battle_state, robot_ids=robot_ids)
        raw_text = self.ask_raw(prompt)
        parsed = self.parse_tasks(raw_text)
        print("[llm_manager] parsed LLM tasks:")
        print(json.dumps(parsed, ensure_ascii=False))
        return parsed

    def parse_tasks(self, text):
        """解析模型文本为 dict；支持从包裹文本中提取 JSON。"""
        if not text:
            raise ValueError("empty LLM response")

        try:
            data = json.loads(text)
        except Exception:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("LLM response does not contain JSON object")
            data = json.loads(text[start : end + 1])

        if not isinstance(data, dict):
            raise ValueError("LLM response must be a dict")
        return data


app = FastAPI(title="LLM Planner Service")
_managers = {}
_config = {}


def _normalize_side(side):
    value = str(side or "").strip().lower()
    if value in ("red", "blue"):
        return value
    return ""


def load_config(config_path):
    """读取 yaml 配置；文件不存在或无效时返回空 dict。"""
    path = str(config_path or "").strip()
    if not path:
        return {}

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print("[llm_manager] load_config failed: {}".format(exc))
        return {}

    return {}


def resolve_model_config(config, side):
    """根据全局/分边配置解析模型参数。"""
    cfg = config if isinstance(config, dict) else {}

    models = cfg.get("models", {})
    if not isinstance(models, dict):
        models = {}

    sides = cfg.get("sides", {})
    if not isinstance(sides, dict):
        sides = {}

    normalized_side = _normalize_side(side)
    side_cfg = sides.get(normalized_side, {}) if normalized_side else {}
    if not isinstance(side_cfg, dict):
        side_cfg = {}

    global_active_model = str(cfg.get("active_model", "")).strip()
    side_active_model = str(side_cfg.get("active_model", "")).strip()
    active_model = side_active_model or global_active_model

    model_cfg = {}
    if active_model and isinstance(models.get(active_model), dict):
        model_cfg = models.get(active_model, {})
    elif not active_model and models:
        first_model_name = sorted(models.keys())[0]
        active_model = str(first_model_name)
        if isinstance(models.get(active_model), dict):
            model_cfg = models.get(active_model, {})

    base_url = str(model_cfg.get("base_url", "")).strip()
    model_name = str(model_cfg.get("model_name", "")).strip()

    try:
        timeout_s = float(model_cfg.get("timeout_s", 30.0))
    except (TypeError, ValueError):
        timeout_s = 30.0

    api_key, api_key_source = _read_api_key_with_source(normalized_side)

    return {
        "active_model": active_model,
        "api_key": api_key,
        "api_key_source": api_key_source,
        "base_url": base_url,
        "model_name": model_name,
        "timeout_s": float(timeout_s),
    }


def _read_api_key_with_source(side):
    normalized_side = _normalize_side(side)
    if normalized_side:
        env_key = "LLM_API_KEY_{}".format(normalized_side.upper())
        scoped_api_key = os.getenv(env_key, "")
        if scoped_api_key:
            return scoped_api_key, env_key

    shared_api_key = os.getenv("LLM_API_KEY", "")
    if shared_api_key:
        return shared_api_key, "LLM_API_KEY"

    shared_api_key = os.getenv("LLM_API", "")
    if shared_api_key:
        return shared_api_key, "LLM_API"

    return "", "MISSING"


# def _read_api_key_by_side(side):
#     api_key, _ = _read_api_key_with_source(side)
#     return api_key


def _get_manager(side=""):
    normalized_side = _normalize_side(side)
    resolved = resolve_model_config(_config, normalized_side)

    if not resolved.get("base_url", ""):
        raise ValueError("missing base_url in config for active model")
    if not resolved.get("model_name", ""):
        raise ValueError("missing model_name in config for active model")

    cache_key = "{}|{}|{}|{}".format(
        normalized_side or "default",
        resolved.get("active_model", ""),
        resolved.get("model_name", ""),
        resolved.get("base_url", ""),
    )
    if cache_key not in _managers:
        print("[llm_manager] init manager: side={}, active_model={}, model_name={}".format(
            normalized_side or "default",
            resolved.get("active_model", ""),
            resolved.get("model_name", ""),
        ))
        _managers[cache_key] = LLMManager(
            api_key=resolved.get("api_key", ""),
            base_url=resolved.get("base_url", ""),
            model=resolved.get("model_name", ""),
            timeout_s=float(resolved.get("timeout_s", 30.0)),
        )
    return _managers[cache_key]


@app.post("/plan")
def plan(payload=Body(default=None)):
    # print("[llm_manager] /plan received payload:")
    # try:
    #     print(json.dumps(payload, ensure_ascii=False))
    # except Exception:
    #     print(str(payload))

    # 宽松模式：接收任意 JSON，优先跑通链路。
    default_side = _normalize_side(os.getenv("LLM_SERVICE_SIDE", ""))
    request_side = ""
    if isinstance(payload, dict):
        battle_state = payload.get("battle_state", payload)
        robot_ids = payload.get("robot_ids", [])
        request_side = payload.get("side", "")
    else:
        battle_state = {"raw_payload": payload}
        robot_ids = []

    if not isinstance(battle_state, dict):
        battle_state = {"raw_battle_state": battle_state}

    if not isinstance(robot_ids, list):
        robot_ids = []
    else:
        robot_ids = [x for x in robot_ids if isinstance(x, str)]

    normalized_side = _normalize_side(request_side)
    if not normalized_side and isinstance(battle_state, dict):
        normalized_side = _normalize_side(battle_state.get("team_color", ""))
    if not normalized_side and robot_ids:
        joined_ids = " ".join(robot_ids).lower()
        if "red" in joined_ids:
            normalized_side = "red"
        elif "blue" in joined_ids:
            normalized_side = "blue"
    if not normalized_side:
        normalized_side = default_side

    print("[llm_manager] normalized request: robot_ids={}, battle_state_keys={}".format(
        robot_ids,
        list(battle_state.keys()) if isinstance(battle_state, dict) else [],
    ))

    try:
        manager = _get_manager(normalized_side)
        result = manager.plan_tasks(battle_state, robot_ids)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _default_config_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, "..", "..", "config", "AI", "llm_config.yaml"))


def main():
    parser = argparse.ArgumentParser(description="LLM Planner Service")
    parser.add_argument("--port", type=int, default=8001, help="FastAPI listening port")
    parser.add_argument("--side", type=str, default="", help="Side identifier (e.g., red or blue)")
    parser.add_argument("--config", type=str, default=_default_config_path(), help="YAML config file path")
    args = parser.parse_args()

    normalized_side = _normalize_side(args.side)

    global _config
    _config = load_config(args.config)
    if not isinstance(_config, dict):
        _config = {}

    if not isinstance(_config.get("models"), dict):
        _config["models"] = {}
    if not isinstance(_config.get("sides"), dict):
        _config["sides"] = {}

    if normalized_side:
        os.environ["LLM_SERVICE_SIDE"] = normalized_side

    resolved_cfg = resolve_model_config(_config, normalized_side)

    print("[llm_manager] Starting LLM Planner Service...")
    print("[llm_manager] Side: {}".format(normalized_side or "unknown"))
    print("[llm_manager] Port: {}".format(args.port))
    print("[llm_manager] Config: {}".format(args.config))
    print("[llm_manager] Active Model: {}".format(resolved_cfg.get("active_model", "")))
    print("[llm_manager] Model Name: {}".format(resolved_cfg.get("model_name", "")))
    print("[llm_manager] Base URL: {}".format(resolved_cfg.get("base_url", "")))
    print("[llm_manager] API Key Source: {}".format(resolved_cfg.get("api_key_source", "MISSING")))
    if resolved_cfg.get("api_key", ""):
        print("[llm_manager] API Key: Provided")
    else:
        print("[llm_manager] API Key: Missing! Please provide via LLM_API_KEY_<SIDE>/LLM_API_KEY/LLM_API.")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
