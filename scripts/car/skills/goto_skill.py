#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import Twist

from skills.base_skill import BaseSkill, RUNNING, SUCCESS, FAILED


class GoToSkill(BaseSkill):
    """
    纯 cmd_vel 导航技能：直接向目标点移动，不依赖 move_base。
    """

    def __init__(self, skill_manager):
        super(GoToSkill, self).__init__(skill_manager)
        self._target_x = 0.0
        self._target_y = 0.0
        self._start_time = None
        self._timeout = 8.0
        self._arrival_tolerance = 0.3
        self._speed = 0.3
        self._angular_speed = 0.8
        self._angle_tolerance = 0.15
        self._has_printed = False

    def start(self, task):
        rospy.loginfo("[GoToSkill] start: received task = %s", task)

        # 判断 task 的类型
        if isinstance(task, dict):
            # 直接从字典中读取 target_x 和 target_y
            self._target_x = float(task.get('target_x', 0.0))
            self._target_y = float(task.get('target_y', 0.0))
            self._timeout = float(task.get('timeout', 8.0))
            rospy.loginfo("[GoToSkill] start: target from dict = (%.2f, %.2f)", self._target_x, self._target_y)
        else:
            # 假设是 TaskCommand 对象（或具有 target_x 属性的对象）
            self._target_x = float(task.target_x)
            self._target_y = float(task.target_y)
            self._timeout = float(getattr(task, 'timeout', 8.0))
            rospy.loginfo("[GoToSkill] start: target from TaskCommand = (%.2f, %.2f)", self._target_x, self._target_y)

        self._start_time = rospy.Time.now().to_sec()
        self._has_printed = False

        self.skill_manager.publish_nav_cancel()

    def update(self):
        elapsed = rospy.Time.now().to_sec() - self._start_time
        if elapsed > self._timeout:
            rospy.logwarn("[GoToSkill] timeout (%.1fs > %.1fs)", elapsed, self._timeout)
            self.skill_manager.publish_stop_velocity()
            return FAILED

        pose = self.skill_manager.get_current_pose()
        if pose is None:
            if not self._has_printed:
                rospy.logwarn("[GoToSkill] waiting for pose...")
                self._has_printed = True
            return RUNNING

        dx = self._target_x - pose.position.x
        dy = self._target_y - pose.position.y
        distance = math.hypot(dx, dy)

        if distance < self._arrival_tolerance:
            rospy.loginfo("[GoToSkill] arrived at target (dist=%.2f)", distance)
            self.skill_manager.publish_stop_velocity()
            return SUCCESS

        angle_to_target = math.atan2(dy, dx)
        current_yaw = self.skill_manager.get_current_yaw()
        if current_yaw is None:
            return RUNNING

        angle_diff = angle_to_target - current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        cmd = Twist()
        if abs(angle_diff) > self._angle_tolerance:
            cmd.linear.x = 0.0
            cmd.angular.z = self._angular_speed * angle_diff
        else:
            cmd.linear.x = self._speed
            cmd.angular.z = 0.0

        self.skill_manager.publish_cmd_vel(cmd)
        return RUNNING

    def stop(self):
        rospy.loginfo("[GoToSkill] stop")
        self.skill_manager.publish_stop_velocity()
        self.skill_manager.publish_nav_cancel()
