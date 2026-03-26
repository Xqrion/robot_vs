#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

from skills.base_skill import BaseSkill, SUCCESS


class StopSkill(BaseSkill):
    """Immediately stop the robot by publishing zero velocity.

    Returns SUCCESS after one tick to signal that the stop command has
    been sent.  The task_engine will keep calling update() while the
    task_engine's current task remains STOP, so the zero-velocity
    command is re-published every tick.
    """

    def start(self):
        self.skill_manager.publish_stop_velocity()
        rospy.loginfo("[%s] StopSkill start", self.skill_manager.ns)

    def update(self):
        self.skill_manager.publish_stop_velocity()
        self._status = SUCCESS
        return self._status

    def stop(self):
        pass
