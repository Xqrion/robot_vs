#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import re
import random

import requests
import rospy

# Kimi (Moonshot AI) API 地址
KIMI_API_URL = "https://api.moonshot.cn/v1/chat/completions"

# 系统提示模板，引导 LLM 返回格式化的 JSON 任务分配
_SYSTEM_PROMPT_TEMPLATE = u"""\
你是 robot_vs 对战游戏的 {team_color}方 战术规划助手。
你的职责是根据当前战场快照，为每辆己方小车规划一条最优行动任务。

【可用动作（action）】
- GOTO  : 导航前往目标坐标 (target.x, target.y)
- STOP  : 原地停车
- ATTACK: 向目标坐标发起攻击

【模式（mode）】
- 0 = 待机   1 = 巡逻   2 = 攻击   3 = 撤退

【战术要点】
1. 血量 < 20 或 弹药 = 0 时，命令小车撤退（mode=3）到安全区域
2. 发现可见敌人（visible_enemies 不为空）时，优先分配小车接近或攻击
3. 无敌人时让小车按不同巡逻点轮换（mode=1）
4. stale=true 表示该机器人状态已过期，对其发送 STOP（mode=0）
5. 多辆小车时应协同配合，避免全部扎堆同一目标

【严格输出格式】（纯 JSON 对象，不要有任何注释或额外文字）
{{
  "<robot_ns>": {{
    "action": "GOTO|STOP|ATTACK",
    "target": {{"x": <float>, "y": <float>}},
    "mode": <0|1|2|3>,
    "reason": "<中文原因>",
    "timeout": <float>
  }}
}}"""


class LLMClient(object):
    """任务规划器：优先调用 Kimi LLM API，失败时自动退回规则引擎。

    输出格式：
    {
      "robot_red_1": {
        "action": "GOTO",
        "target": {"x": 1.0, "y": 2.0},
        "mode": 1,
        "reason": "occupy point"
      }
    }
    """

    def __init__(self, planner_fn=None, patrol_points=None, flank_offset=0.1,
                 enabled=False, api_key="", model="moonshot-v1-8k", timeout_s=10.0):
        self._planner_fn = planner_fn
        self._patrol_points = patrol_points or [
            {"x": 1.5, "y": 0.0},
            {"x": 0.0, "y": 0.0},
            {"x": 1.5, "y": 1.5},
            {"x": 0.0, "y": 1.5},# zai yaml li mian qu shezhi!!!!
        ]
        self._flank_offset = float(flank_offset)
        self._patrol_hold_s = 3.0
        # 无敌人场景下的逐车巡逻状态。
        # {robot_id: {"idx": int, "hold_until": float, "last_success_task_id": int}}
        self._patrol_state = {}

        # Kimi LLM 配置
        self._enabled = bool(enabled)
        # api_key 优先使用构造参数，其次读取环境变量 MOONSHOT_API_KEY
        self._api_key = str(api_key).strip() if api_key else os.environ.get("MOONSHOT_API_KEY", "")
        self._model = str(model) if model else "moonshot-v1-8k"
        self._timeout_s = float(timeout_s) if timeout_s and float(timeout_s) > 0 else 10.0

        if self._enabled and self._api_key:
            rospy.loginfo("[LLMClient] Kimi LLM enabled: model=%s timeout=%.1fs", self._model, self._timeout_s)
        elif self._enabled:
            rospy.logwarn(
                "[LLMClient] LLM enabled but api_key is empty — falling back to rule-based planner. "
                "Set ~llm/api_key in YAML or MOONSHOT_API_KEY env var."
            )

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def plan_tasks(self, battle_state):
        """根据战场状态规划任务。

        优先调用 Kimi LLM API（当 enabled=True 且 api_key 有效时）；
        LLM 调用失败或未启用时，自动退回以下规则引擎：
        1) 机器人状态缺失或过期 -> STOP
        2) 机器人死亡 -> STOP
        3) 任务失败 -> 重试巡逻点
        4) 发现可见敌人且弹药充足 -> ATTACK/GOTO
        5) 血量低或弹药低 -> 撤退（GOTO 安全点，mode=3）
        6) 其他情况 -> 巡逻
        """
        if self._planner_fn is not None:
            return self._planner_fn(battle_state)

        # 尝试调用 Kimi LLM
        if self._enabled and self._api_key:
            try:
                tasks = self._plan_with_llm(battle_state)
                rospy.logdebug("[LLMClient] Kimi LLM planning succeeded: %s", list(tasks.keys()))
                return tasks
            except Exception as exc:
                rospy.logwarn("[LLMClient] Kimi API failed, falling back to rule-based: %s", exc)

        # 规则引擎兜底
        if not isinstance(battle_state, dict):
            return {}

        robot_ids, friendly = self._extract_friendly_robots(battle_state)
        if not robot_ids:
            return {}

        visible_enemies = self._extract_visible_enemies(battle_state)
        tasks = {}
        for idx, robot_id in enumerate(robot_ids):
            state_entry = friendly.get(robot_id, {})
            tasks[robot_id] = self._plan_single_robot_task(robot_id, state_entry, idx, visible_enemies)

        return tasks

    def _plan_single_robot_task(self, robot_id, state_entry, robot_index, visible_enemies):
        if self._is_robot_data_missing(state_entry):
            return self._stop_task("missing/stale robot state", timeout=1.5)

        state = state_entry.get("state")
        alive = bool(self._read_value(state, "alive", True))
        hp = self._to_float(self._read_value(state, "hp", 100.0), 100.0)
        ammo = self._to_float(self._read_value(state, "ammo", 0.0), 0.0)
        in_combat = bool(self._read_value(state, "in_combat", False))
        task_status = str(self._read_value(state, "task_status", "")).upper()
        current_action = str(self._read_value(state, "current_action", "")).upper()

        if (not alive) or hp <= 0.0:
            return self._stop_task("robot not alive", timeout=5.0)

        if task_status in ("FAILED", "ABORTED", "TIMEOUT"):
            retry_point = self._get_patrol_point(robot_id, robot_index)
            return self._build_task(
                action="GOTO",
                target=retry_point,
                mode=1,
                reason="retry after task failure",
                timeout=4.0,
            )

        # 资源不足时优先保命。
        if hp < 20.0 or ammo <= 0.0:
            return self._build_task(
                action="GOTO",
                target=self._get_safe_point(robot_id, robot_index),
                mode=3,
                reason="retreat (low hp/ammo)",
                timeout=6.0,
            )

        if visible_enemies:
            enemy = visible_enemies[robot_index % len(visible_enemies)]
            enemy_x = self._to_float(enemy.get("x", 0.0), 0.0)
            enemy_y = self._to_float(enemy.get("y", 0.0), 0.0)
            engage_target = self._random_near_enemy_point(enemy_x, enemy_y)

            if in_combat or current_action == "ATTACK":
                return self._build_task(
                    action="ATTACK",
                    target=engage_target,
                    mode=2,
                    reason="engage visible enemy",
                    timeout=2.0,
                )

            if robot_index % 2 == 0:
                return self._build_task(
                    action="GOTO",
                    target=engage_target,
                    mode=2,
                    reason="contain visible enemy",
                    timeout=4.0,
                )

            flank_y = enemy_y + self._flank_offset if ((robot_index // 2) % 2 == 0) else enemy_y - self._flank_offset
            flank_target = self._random_near_enemy_point(enemy_x + self._flank_offset, flank_y)
            return self._build_task(
                action="GOTO",
                target=flank_target,
                mode=2,
                reason="flank visible enemy",
                timeout=4.0,
            )

        if ammo <= 5.0:
            return self._build_task(
                action="GOTO",
                target=self._get_safe_point(robot_id, robot_index),
                mode=3,
                reason="retreat (low ammo)",
                timeout=5.0,
            )

        patrol_transition_task = self._next_patrol_task_if_ready(
            robot_id=robot_id,
            robot_index=robot_index,
            state=state,
            task_status=task_status,
            current_action=current_action,
        )
        if patrol_transition_task is not None:
            return patrol_transition_task

        return self._build_task(
            action="GOTO",
            target=self._get_patrol_point(robot_id, robot_index),
            mode=1,
            reason="patrol (no visible enemy)",
            timeout=12.0, #时间尽量大一些 
        )

    def _next_patrol_task_if_ready(self, robot_id, robot_index, state, task_status, current_action):
        patrol_state = self._get_patrol_state(robot_id, robot_index)
        now = self._now()

        current_task_id = int(self._to_float(self._read_value(state, "current_task_id", 0), 0))
        last_success_task_id = int(patrol_state.get("last_success_task_id", -1))

        # GOTO 成功后先短暂停留，再切换到下一个巡逻点。
        if task_status == "SUCCESS" and current_action == "GOTO" and current_task_id != last_success_task_id:
            patrol_state["last_success_task_id"] = current_task_id
            patrol_state["hold_until"] = now + self._patrol_hold_s
            return self._build_task(
                action="STOP",
                target={"x": 0.0, "y": 0.0},
                mode=1,
                reason="patrol SUCCESS and hold before next waypoint",
                timeout=self._patrol_hold_s,
            )

        hold_until = float(patrol_state.get("hold_until", 0.0) or 0.0)
        if hold_until > now:
            return self._build_task(
                action="STOP",
                target={"x": 0.0, "y": 0.0},
                mode=1,
                reason="patrol hold before next waypoint",
                timeout=max(0.5, hold_until - now),
            )

        if hold_until > 0.0 and hold_until <= now:
            patrol_state["hold_until"] = 0.0
            if self._patrol_points:
                patrol_state["idx"] = (int(patrol_state.get("idx", 0)) + 1) % len(self._patrol_points)

        return None

    def _extract_friendly_robots(self, battle_state):
        context = self._extract_context(battle_state)
        friendly = context.get("friendly", {})
        if isinstance(friendly, dict) and friendly:
            return sorted(friendly.keys()), friendly

        # 兼容旧输入格式 {"my_cars": [...]}。
        my_cars = battle_state.get("my_cars", [])
        if isinstance(my_cars, list) and my_cars:
            generated = {}
            for ns in my_cars:
                generated[ns] = {"state": {}, "stale": False}
            return list(my_cars), generated

        return [], {}

    def _extract_visible_enemies(self, battle_state):
        context = self._extract_context(battle_state)
        enemy_block = context.get("enemy", {})
        if not isinstance(enemy_block, dict):
            return []

        if enemy_block.get("stale", True):
            return []

        state = enemy_block.get("state")
        if not isinstance(state, dict):
            return []

        # 推荐格式：{"visible_enemies": [{"x":..,"y":..}, ...]}
        visible = state.get("visible_enemies")
        if isinstance(visible, list) and visible:
            return [e for e in visible if isinstance(e, dict)]

        # 兼容格式：{"enemies": [{"x":..,"y":..,"visible":true}, ...]}
        enemies = state.get("enemies")
        if isinstance(enemies, list):
            result = []
            for enemy in enemies:
                if isinstance(enemy, dict) and enemy.get("visible", True):
                    result.append(enemy)
            return result

        # 最小格式：state 本身包含 x/y，且 visible 可选。
        if "x" in state and "y" in state and state.get("visible", True):
            return [state]

        return []

    def _extract_context(self, planner_input):
        # A) planner_input 顶层直接包含 friendly/enemy
        if isinstance(planner_input.get("friendly"), dict) or isinstance(planner_input.get("enemy"), dict):
            return {
                "friendly": planner_input.get("friendly", {}),
                "enemy": planner_input.get("enemy", {}),
            }

        # B) planner_input 里包含 battle_state，且其内含 friendly/enemy
        nested = planner_input.get("battle_state", {})
        if isinstance(nested, dict):
            return {
                "friendly": nested.get("friendly", {}),
                "enemy": nested.get("enemy", {}),
            }

        return {"friendly": {}, "enemy": {}}

    def _is_robot_data_missing(self, state_entry):
        if not isinstance(state_entry, dict):
            return True
        if state_entry.get("stale", True):
            return True
        if state_entry.get("state") is None:
            return True
        return False

    def _build_task(self, action, target, mode, reason, timeout):
        target_point = self._normalize_patrol_point(target)
        return {
            "action": str(action),
            "target": {
                "x": float(target_point["x"]),
                "y": float(target_point["y"]),
            },
            "mode": int(mode),
            "reason": str(reason),
            "timeout": float(timeout),
        }

    def _stop_task(self, reason, timeout=2.0):
        return self._build_task(
            action="STOP",
            target={"x": 0.0, "y": 0.0},
            mode=0,
            reason=reason,
            timeout=timeout,
        )

    def _get_patrol_point(self, robot_id, robot_index):
        if not self._patrol_points:
            return {"x": 0.0, "y": 0.0}

        patrol_state = self._get_patrol_state(robot_id, robot_index)
        idx = int(patrol_state.get("idx", 0)) % len(self._patrol_points)
        return self._normalize_patrol_point(self._patrol_points[idx])

    def _get_safe_point(self, robot_id, robot_index):
        patrol = self._get_patrol_point(robot_id, robot_index)
        return {
            "x": patrol["x"] - self._flank_offset,
            "y": patrol["y"] - self._flank_offset,
        }

    def _get_patrol_state(self, robot_id, robot_index):
        if robot_id not in self._patrol_state:
            seed_idx = robot_index
            if self._patrol_points:
                seed_idx = robot_index % len(self._patrol_points)
            self._patrol_state[robot_id] = {
                "idx": seed_idx,
                "hold_until": 0.0,
                "last_success_task_id": -1,
            }
        return self._patrol_state[robot_id]

    def _now(self):
        return rospy.Time.now().to_sec()

    def _read_value(self, obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        try:
            return getattr(obj, key)
        except Exception:
            return default

    def _to_float(self, value, default):
        try:
            return float(value)
        except Exception:
            return float(default)

    def _normalize_patrol_point(self, point):
        if isinstance(point, dict):
            return {
                "x": float(point.get("x", 0.0)),
                "y": float(point.get("y", 0.0)),
            }

        if isinstance(point, (list, tuple)) and len(point) >= 2:
            return {
                "x": float(point[0]),
                "y": float(point[1]),
            }

        return {"x": 0.0, "y": 0.0}

    def _random_near_enemy_point(self, x, y):
        # 在敌人附近生成随机目标点，避免直接重合导致拥挤/碰撞。
        radius_min = max(0.05, self._flank_offset * 0.8)
        radius_max = max(radius_min + 0.05, self._flank_offset * 2.5)
        theta = random.uniform(0.0, 2.0 * 3.1415926)
        radius = random.uniform(radius_min, radius_max)
        return {
            "x": float(x) + radius * math.cos(theta),
            "y": float(y) + radius * math.sin(theta),
        }

    # ------------------------------------------------------------------
    # Kimi LLM 集成
    # ------------------------------------------------------------------

    def _plan_with_llm(self, battle_state):
        """调用 Kimi API 获取任务规划，返回任务字典。

        抛出异常时由调用方捕获并退回规则引擎。
        """
        team_color = battle_state.get("team_color", "unknown") if isinstance(battle_state, dict) else "unknown"
        system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(team_color=team_color)
        user_message = json.dumps(battle_state, ensure_ascii=False, indent=2, default=str)
        raw_text = self._call_remote_llm(system_prompt, user_message)
        tasks = self._parse_llm_json(raw_text)
        return tasks

    def _call_remote_llm(self, system_prompt, user_message):
        """调用 Kimi (Moonshot AI) Chat Completions API，返回模型原始回复文本。

        参数：
            system_prompt  -- 系统提示（角色设定 + 输出格式要求）
            user_message   -- 用户消息（战场快照 JSON 字符串）

        返回：
            模型回复的字符串内容

        抛出：
            requests.HTTPError / requests.Timeout / ValueError 等
        """
        headers = {
            "Authorization": "Bearer {}".format(self._api_key),
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.3,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(
            KIMI_API_URL,
            headers=headers,
            json=payload,
            timeout=self._timeout_s,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _parse_llm_json(self, text):
        """解析 LLM 返回的 JSON 文本，支持从 Markdown 代码块中提取。"""
        if not text:
            raise ValueError("empty LLM response")

        # 尝试从 Markdown 代码块中提取 JSON（```json ... ``` 或 ``` ... ```）
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            text = code_block.group(1)

        data = json.loads(text.strip())
        if not isinstance(data, dict):
            raise ValueError("LLM response must be a JSON object (dict), got: {}".format(type(data)))
        return data
