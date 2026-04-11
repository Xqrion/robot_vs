#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy

from skills.base_skill import BaseSkill, SUCCESS


class StopSkill(BaseSkill):
    """通过取消导航目标并发布零速度来停止机器人。

    该技能会发起真正的 move_base 目标取消，而不是把当前位姿当作新目标重设。
    """

    def start(self, params=None):
        # 取消当前导航目标
        self.skill_manager.cancel_nav_goal()

        # 发布零速度，确保机器人立即停止
        self.skill_manager.publish_stop_velocity()

        rospy.loginfo("[%s] StopSkill start: navigation cancelled", self.skill_manager.ns)

    def update(self):
        self._status = SUCCESS
        return self._status

    def stop(self):
        pass
