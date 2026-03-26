#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json


class LLMClient(object):
    """Rule-based planner (Mock LLM).

    Output format:
    {
      "robot_red_1": {
        "action": "GOTO",
        "target": {"x": 1.0, "y": 2.0},
        "mode": 1,
        "reason": "occupy point"
      }
    }
    """

    def __init__(self, planner_fn=None, patrol_points=None, flank_offset=1.0):
        self._planner_fn = planner_fn
        self._patrol_points = patrol_points or [
            {"x": 0.0, "y": 0.0},
            {"x": 1.5, "y": 0.0},
            {"x": 1.5, "y": 1.5},
            {"x": 0.0, "y": 1.5},
        ]
        self._flank_offset = float(flank_offset)

    def plan_tasks(self, battle_state):
        """Plan tasks from battle state using deterministic rules.

        Rules (test-friendly):
        1) stale/missing robot state -> STOP
        2) dead robot -> STOP
        3) failed task -> retry patrol point
        4) visible enemy + enough ammo -> ATTACK/GOTO
        5) low hp or low ammo -> RETREAT (GOTO safe point, mode=3)
        6) otherwise -> patrol
        """
        if self._planner_fn is not None:
            return self._planner_fn(battle_state)

        if not isinstance(battle_state, dict):
            return {}

        robot_ids, friendly = self._extract_friendly_robots(battle_state)
        if not robot_ids:
            return {}

        visible_enemies = self._extract_visible_enemies(battle_state)
        tasks = {}
        for idx, robot_id in enumerate(robot_ids):
            state_entry = friendly.get(robot_id, {})
            tasks[robot_id] = self._plan_single_robot_task(state_entry, idx, visible_enemies)

        return tasks

    def _plan_single_robot_task(self, state_entry, robot_index, visible_enemies):
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
            retry_point = self._get_patrol_point(robot_index)
            return self._build_task(
                action="GOTO",
                target=retry_point,
                mode=1,
                reason="retry after task failure",
                timeout=4.0,
            )

        # Low resources first: keep robot safe.
        if hp < 20.0 or ammo <= 0.0:
            return self._build_task(
                action="GOTO",
                target=self._get_safe_point(robot_index),
                mode=3,
                reason="retreat (low hp/ammo)",
                timeout=6.0,
            )

        if visible_enemies:
            enemy = visible_enemies[robot_index % len(visible_enemies)]
            enemy_x = self._to_float(enemy.get("x", 0.0), 0.0)
            enemy_y = self._to_float(enemy.get("y", 0.0), 0.0)

            if in_combat or current_action == "ATTACK":
                return self._build_task(
                    action="ATTACK",
                    target={"x": enemy_x, "y": enemy_y},
                    mode=2,
                    reason="engage visible enemy",
                    timeout=2.0,
                )

            if robot_index % 2 == 0:
                return self._build_task(
                    action="GOTO",
                    target={"x": enemy_x, "y": enemy_y},
                    mode=2,
                    reason="contain visible enemy",
                    timeout=4.0,
                )

            flank_y = enemy_y + self._flank_offset if ((robot_index // 2) % 2 == 0) else enemy_y - self._flank_offset
            return self._build_task(
                action="GOTO",
                target={"x": enemy_x + self._flank_offset, "y": flank_y},
                mode=2,
                reason="flank visible enemy",
                timeout=4.0,
            )

        if ammo <= 5.0:
            return self._build_task(
                action="GOTO",
                target=self._get_safe_point(robot_index),
                mode=3,
                reason="retreat (low ammo)",
                timeout=5.0,
            )

        return self._build_task(
            action="GOTO",
            target=self._get_patrol_point(robot_index),
            mode=1,
            reason="patrol (no visible enemy)",
            timeout=8.0,
        )

    def _extract_friendly_robots(self, battle_state):
        context = self._extract_context(battle_state)
        friendly = context.get("friendly", {})
        if isinstance(friendly, dict) and friendly:
            return sorted(friendly.keys()), friendly

        # Compatible fallback for earlier input style {"my_cars": [...]}.
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

        # Preferred shape: {"visible_enemies": [{"x":..,"y":..}, ...]}
        visible = state.get("visible_enemies")
        if isinstance(visible, list) and visible:
            return [e for e in visible if isinstance(e, dict)]

        # Compatible shape: {"enemies": [{"x":..,"y":..,"visible":true}, ...]}
        enemies = state.get("enemies")
        if isinstance(enemies, list):
            result = []
            for enemy in enemies:
                if isinstance(enemy, dict) and enemy.get("visible", True):
                    result.append(enemy)
            return result

        # Minimal shape: state itself contains x/y and optional visible flag.
        if "x" in state and "y" in state and state.get("visible", True):
            return [state]

        return []

    def _extract_context(self, planner_input):
        # A) planner_input has top-level friendly/enemy
        if isinstance(planner_input.get("friendly"), dict) or isinstance(planner_input.get("enemy"), dict):
            return {
                "friendly": planner_input.get("friendly", {}),
                "enemy": planner_input.get("enemy", {}),
            }

        # B) planner_input has battle_state with friendly/enemy inside
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

    def _get_patrol_point(self, robot_index):
        if not self._patrol_points:
            return {"x": 0.0, "y": 0.0}
        return self._normalize_patrol_point(self._patrol_points[robot_index % len(self._patrol_points)])

    def _get_safe_point(self, robot_index):
        patrol = self._get_patrol_point(robot_index)
        return {
            "x": patrol["x"] - self._flank_offset,
            "y": patrol["y"] - self._flank_offset,
        }

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

    def _call_remote_llm(self, prompt):
        """Reserved: call real remote LLM API and return raw text."""
        raise NotImplementedError("Remote LLM API is not connected yet")

    def _parse_llm_json(self, text):
        """Reserved: parse JSON text returned by remote LLM."""
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("LLM response must be a dict")
        return data
