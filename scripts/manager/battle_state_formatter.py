#!/usr/bin/env python
# -*- coding: utf-8 -*-


class BattleStateFormatter(object):
    """Converts raw battle state to planner input."""

    def build(self, battle_state, team_color, my_cars):
        if battle_state is None:
            battle_state = {}

        friendly = battle_state.get("friendly", {}) if isinstance(battle_state, dict) else {}
        enemy = battle_state.get("enemy", {}) if isinstance(battle_state, dict) else {}

        return {
            "team_color": str(team_color),
            "my_cars": list(my_cars),
            "battle_state": battle_state,
            "friendly": friendly,
            "enemy": enemy,
        }
