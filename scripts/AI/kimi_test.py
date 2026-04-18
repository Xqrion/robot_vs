import json
import os

from openai import OpenAI


def build_prompt(battle_state, robot_ids):
    payload = {
        "battle_state": battle_state,
        "robot_ids": robot_ids,
        "required_output": {
            "<robot_id>": {
                "action": "STOP | GOTO | ATTACK | ROTATE",
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
    print("[kimi_manager] built prompt:")
    print(json.dumps(payload, ensure_ascii=False))
    return json.dumps(payload, ensure_ascii=False)


def parse_tasks(text):
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


def simple():
    
    # 简单测试一下 OpenAI
    client = OpenAI(api_key=os.getenv("KIMI_API_KEY", ""), base_url="https://api.moonshot.cn/v1")
    response = client.chat.completions.create(
        model="kimi-k2.5",
        messages=[{"role": "user", "content": "你好"}],
        temperature=1.0  # 改成 1
    )
    print("✅ Kimi 回复：", response.choices[0].message.content)

def main():
    api_key = os.getenv("TEST_API_KEY", "")
    if not api_key:
        raise ValueError("TEST_API_KEY is empty")

    base_url = os.getenv("TEST_BASE_URL", "https://api.moonshot.cn/v1")
    client = OpenAI(api_key=api_key, base_url=base_url)

    # 测试输入与 kimi_manager 保持一致的结构。
    battle_state = {
        "friendly": {
            "robot_red": {
                "state": {"hp": 100, "ammo": 50, "alive": True, "task_status": "IDLE"},
                "stale": False,
            }
        },
        "enemy": {
            "state": {"visible_enemies": [{"x": 1.0, "y": 2.0}]},
            "stale": False,
        },
    }
    robot_ids = ["robot_red"]
    prompt = build_prompt(battle_state, robot_ids)

    response = client.chat.completions.create(
        # model="kimi-k2.5",
        # model="kimi-k2-turbo-preview",
        # model="glm-4.7-flash",
        model="glm-4.7-flashX",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是多机器人战术规划器。"
                    "你必须只输出一个 JSON 对象，不能输出解释、markdown 或额外文本。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        timeout=30.0,
        extra_body={
        "enable_thinking": False
        },
    )

    raw_text = str(response.choices[0].message.content or "").strip()
    print("[kimi_test] raw LLM response:")
    print(raw_text)

    parsed = parse_tasks(raw_text)
    print("[kimi_test] parsed tasks JSON:")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    #simple()
    main()
