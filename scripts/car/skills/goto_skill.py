#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseStamped, Quaternion

from skills.base_skill import BaseSkill, RUNNING, SUCCESS, FAILED


class GoToSkill(BaseSkill):
    """Navigate to a (x, y) target using move_base_simple/goal.

    Completes with SUCCESS when move_base reports SUCCEEDED (status == 3).
    Completes with FAILED when move_base reports any failure status.
    """

    # move_base GoalStatus values
    _STATUS_SUCCEEDED = 3
    _STATUS_FAILED = {4, 5, 8, 9}  # ABORTED, REJECTED, LOST, etc.

    def __init__(self, skill_manager, target_x, target_y, frame_id="map"):
        super(GoToSkill, self).__init__(skill_manager)
        self.target_x = float(target_x)
        self.target_y = float(target_y)
        self.frame_id = str(frame_id)

    def start(self):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = self.frame_id
        goal.pose.position.x = self.target_x
        goal.pose.position.y = self.target_y
        goal.pose.position.z = 0.0
        goal.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        self.skill_manager.reset_nav_status()
        self.skill_manager.publish_nav_goal(goal)
        self._status = RUNNING

        rospy.loginfo(
            "[%s] GoToSkill start: target=(%.2f, %.2f)",
            self.skill_manager.ns, self.target_x, self.target_y,
        )

    def update(self):
        nav_status = self.skill_manager.nav_status_code

        if nav_status == self._STATUS_SUCCEEDED:
            self._status = SUCCESS
        elif nav_status in self._STATUS_FAILED:
            rospy.logwarn(
                "[%s] GoToSkill failed: move_base status=%d",
                self.skill_manager.ns, nav_status,
            )
            self._status = FAILED
        else:
            self._status = RUNNING

        return self._status

    def stop(self):
        # Cancel by publishing zero velocity; move_base will idle.
        self.skill_manager.publish_stop_velocity()
