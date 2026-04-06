#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import json

class BattleStateFormatter(object):
    """将原始战场状态转换为规划器输入。"""

    def build(self, battle_state, team_color, my_cars):
        if battle_state is None:
            battle_state = {}

        friendly = battle_state.get("friendly", {}) if isinstance(battle_state, dict) else {}
        enemy = battle_state.get("enemy", {}) if isinstance(battle_state, dict) else {}

        # rospy.loginfo("ZHONGJIAN from global observer to formatter=%s", json.dumps(battle_state, ensure_ascii=False, indent=2, default=str))

        battle_state_format = {
            "team_color": str(team_color),
            "my_cars": list(my_cars),
            "friendly": friendly,
            "enemy": enemy,
        }

        # rospy.loginfo("ZHONGJIAN from formatter to LLM=%s", json.dumps(battle_state_format, ensure_ascii=False, indent=2, default=str))

        return battle_state_format
