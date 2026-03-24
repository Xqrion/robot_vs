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

        Rules:
        1) Visible enemy: split robots by index, even->contain, odd->flank.
        2) No enemy: patrol preset points.
        3) Missing data: STOP with reason.
        """
        if self._planner_fn is not None:
            return self._planner_fn(battle_state)

        if not isinstance(battle_state, dict):
            return {}

        robot_ids, friendly = self._extract_friendly_robots(battle_state)
        if not robot_ids:
            return {}

        tasks = {}
        visible_enemies = self._extract_visible_enemies(battle_state)

        if visible_enemies:
            primary_enemy = visible_enemies[0]
            enemy_x = float(primary_enemy.get("x", 0.0))
            enemy_y = float(primary_enemy.get("y", 0.0))

            for idx, robot_id in enumerate(robot_ids):
                state_entry = friendly.get(robot_id, {})
                if self._is_robot_data_missing(state_entry):
                    tasks[robot_id] = self._stop_task("missing/stale robot state")
                    continue

                if idx % 2 == 0:
                    tasks[robot_id] = {
                        "action": "GOTO",
                        "target": {"x": enemy_x, "y": enemy_y},
                        "mode": 2,
                        "reason": "contain visible enemy",
                    }
                else:
                    flank_y = enemy_y + self._flank_offset if ((idx // 2) % 2 == 0) else enemy_y - self._flank_offset
                    tasks[robot_id] = {
                        "action": "GOTO",
                        "target": {"x": enemy_x + self._flank_offset, "y": flank_y},
                        "mode": 2,
                        "reason": "flank visible enemy",
                    }

            return tasks

        # No visible enemy -> patrol
        for idx, robot_id in enumerate(robot_ids):
            state_entry = friendly.get(robot_id, {})
            if self._is_robot_data_missing(state_entry):
                tasks[robot_id] = self._stop_task("missing/stale robot state")
                continue

            point = self._normalize_patrol_point(self._patrol_points[idx % len(self._patrol_points)])
            tasks[robot_id] = {
                "action": "GOTO",
                "target": {
                    "x": float(point["x"]),
                    "y": float(point["y"]),
                },
                "mode": 1,
                "reason": "patrol (no visible enemy)",
            }

        return tasks

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

    def _stop_task(self, reason):
        return {
            "action": "STOP",
            "target": {"x": 0.0, "y": 0.0},
            "mode": 0,
            "reason": reason,
        }

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
