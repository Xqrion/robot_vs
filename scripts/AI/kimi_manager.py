### 在miniconda的robotvs下运行

import argparse
import json
import os
import time

from fastapi import Body, FastAPI, HTTPException
from openai import OpenAI
import uvicorn


class KimiManager(object):
    """Kimi API 适配器：负责构造提示词并返回可执行任务 JSON。"""

    def __init__(self, api_key=None, base_url=None, model="kimi-k2-turbo-preview", timeout_s=60.0):
        self.api_key = str(api_key or self._read_api_key())
        self.base_url = str(base_url or self._read_base_url())
        self.model = str(model)
        self.timeout_s = float(timeout_s)

        if not self.api_key:
            raise ValueError("Kimi API key is empty. Set env KIMI_API_KEY")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _read_api_key(self):
        return os.getenv("KIMI_API_KEY", "")

    def _read_base_url(self):
        return os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")

    def ask_raw(self, prompt):
        """向 Kimi 发送纯文本 prompt，返回模型原始文本。"""
        # print("[kimi_manager] sending prompt to LLM:")
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
        )
        request_end_s = time.time()
        request_elapsed_s = request_end_s - request_start_s
        request_time_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(request_end_s))
        raw_text = str(response.choices[0].message.content or "").strip()
        print("[kimi_manager] raw LLM response:")
        print(raw_text)
        print("[kimi_manager] request_time={}, elapsed_s={:.3f}".format(
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
                    "action": "STOP | GOTO | ATTACK",
                    "target": {"x": 0.0, "y": 0.0},
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
        print("[kimi_manager] parsed LLM tasks:")
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


app = FastAPI(title="Kimi Planner Service")
_managers = {}


def _normalize_side(side):
    value = str(side or "").strip().lower()
    if value in ("red", "blue"):
        return value
    return ""


def _read_api_key_with_source(side):
    normalized_side = _normalize_side(side)
    if normalized_side:
        env_key = "KIMI_API_KEY_{}".format(normalized_side.upper())
        scoped_api_key = os.getenv(env_key, "")
        if scoped_api_key:
            return scoped_api_key, env_key

    shared_api_key = os.getenv("KIMI_API_KEY", "")
    if shared_api_key:
        return shared_api_key, "KIMI_API_KEY"

    return "", "MISSING"


def _read_api_key_by_side(side):
    api_key, _ = _read_api_key_with_source(side)
    return api_key


def _get_manager(side=""):
    normalized_side = _normalize_side(side)
    cache_key = normalized_side or "default"
    if cache_key not in _managers:
        api_key, api_key_source = _read_api_key_with_source(normalized_side)
        print("[kimi_manager] init manager: side={}, api_key_source={}".format(
            normalized_side or "default",
            api_key_source,
        ))
        _managers[cache_key] = KimiManager(api_key=api_key)
    return _managers[cache_key]


@app.post("/plan")
def plan(payload=Body(default=None)):
    # print("[kimi_manager] /plan received payload:")
    # try:
    #     print(json.dumps(payload, ensure_ascii=False))
    # except Exception:
    #     print(str(payload))

    # 宽松模式：接收任意 JSON，优先跑通链路。
    default_side = _normalize_side(os.getenv("KIMI_SERVICE_SIDE", ""))
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

    # print("[kimi_manager] normalized request: robot_ids={}, battle_state_keys={}".format(
    #     robot_ids,
    #     list(battle_state.keys()) if isinstance(battle_state, dict) else [],
    # ))

    try:
        manager = _get_manager(normalized_side)
        result = manager.plan_tasks(battle_state, robot_ids)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    parser = argparse.ArgumentParser(description="Kimi Planner Service")
    parser.add_argument("--port", type=int, default=8001, help="FastAPI listening port")
    parser.add_argument("--api-key", type=str, default="", help="Kimi API Key")
    parser.add_argument("--side", type=str, default="", help="Side identifier (e.g., red or blue)")
    args = parser.parse_args()

    normalized_side = _normalize_side(args.side)
    api_key_source = "--api-key"

    # 如果命令行没有提供 api-key，但提供了 side，则尝试从环境变量读取
    if not args.api_key:
        args.api_key, api_key_source = _read_api_key_with_source(normalized_side)

    # 设置回环境变量，这样 KimiManager 内部初始化时如果不传 api_key，就会读到这个值
    if args.api_key:
        if normalized_side:
            os.environ["KIMI_API_KEY_{}".format(normalized_side.upper())] = args.api_key
            os.environ["KIMI_SERVICE_SIDE"] = normalized_side
        os.environ["KIMI_API_KEY"] = args.api_key
    elif normalized_side:
        os.environ["KIMI_SERVICE_SIDE"] = normalized_side

    print(f"[kimi_manager] Starting Kimi Planner Service...")
    print(f"[kimi_manager] Side: {normalized_side or 'unknown'}")
    print(f"[kimi_manager] Port: {args.port}")
    if args.api_key:
        print("[kimi_manager] API Key: Provided")
        if api_key_source == "--api-key":
            print("[kimi_manager] API Key Source: --api-key")
        elif api_key_source == "KIMI_API_KEY":
            print("[kimi_manager] API Key Source: fallback shared env KIMI_API_KEY")
        else:
            print("[kimi_manager] API Key Source: side env {}".format(api_key_source))
    else:
        print("[kimi_manager] API Key: Missing! Please provide via --api-key or env vars.")

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()