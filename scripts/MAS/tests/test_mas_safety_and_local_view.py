#!/usr/bin/env python3

import os
import sys
import unittest


MAS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if MAS_ROOT not in sys.path:
    sys.path.insert(0, MAS_ROOT)

from agents.car_agent import CarAgent  # noqa: E402
from mas_manager import _build_local_state_by_robot  # noqa: E402


class _DummyLLMClient(object):
    async def request_actions(self, messages, profile, trace_tag=""):
        return []


class CarAgentNormalizationTests(unittest.TestCase):
    def test_normalize_task_clamps_xy_to_map_bounds(self):
        agent = CarAgent(
            robot_id="robot_red_1",
            llm_client=_DummyLLMClient(),
            models_cfg={"llm": {}, "car_model": {"name": "dummy"}, "leader_model": {"name": "dummy"}},
            prompts_cfg={},
        )

        task = agent._normalize_task(
            {
                "action": "GOTO",
                "target": {"x": 100.0, "y": -100.0, "yaw": 0.5},
                "mode": 1,
                "timeout": 3.0,
            }
        )

        self.assertEqual(task["target"]["x"], 3.8)
        self.assertEqual(task["target"]["y"], -1.8)
        self.assertEqual(task["target"]["yaw"], 0.5)

    def test_normalize_task_respects_runtime_map_bounds_override(self):
        agent = CarAgent(
            robot_id="robot_red_1",
            llm_client=_DummyLLMClient(),
            models_cfg={
                "llm": {},
                "car_model": {"name": "dummy"},
                "leader_model": {"name": "dummy"},
                "runtime": {"map_bounds": {"x_min": -2.0, "x_max": 2.0, "y_min": -1.0, "y_max": 1.0}},
            },
            prompts_cfg={},
        )

        task = agent._normalize_task({"action": "GOTO", "target": {"x": 10.0, "y": -10.0}})
        self.assertEqual(task["target"]["x"], 2.0)
        self.assertEqual(task["target"]["y"], -1.0)


class LocalStateBuilderTests(unittest.TestCase):
    def test_local_visible_enemies_prefer_robot_specific_view(self):
        battle_state = {
            "friendly": {
                "robot_red_1": {
                    "state": {
                        "hp": 80,
                        "ammo": 20,
                        "visible_enemies": [{"x": 1.0, "y": 0.1}],
                    }
                },
                "robot_red_2": {
                    "state": {
                        "hp": 90,
                        "ammo": 18,
                    }
                },
            },
            "enemy": {"state": {"visible_enemies": [{"x": 3.0, "y": 1.0}]}},
        }

        local = _build_local_state_by_robot(
            side="red",
            battle_state=battle_state,
            robot_ids=["robot_red_1", "robot_red_2"],
        )

        self.assertEqual(local["robot_red_1"]["visible_enemies"], [{"x": 1.0, "y": 0.1}])
        self.assertEqual(local["robot_red_2"]["visible_enemies"], [{"x": 3.0, "y": 1.0}])


if __name__ == "__main__":
    unittest.main()
