#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from skills.base_skill import BaseSkill, RUNNING, SUCCESS, FAILED


class RotateSkill(BaseSkill):
    """原地旋转到指定 target_yaw。"""

    def __init__(self, skill_manager):
        super(RotateSkill, self).__init__(skill_manager)
        self.target_yaw = 0.0
        self.yaw_tolerance = 0.05
        self.angular_speed = 0.6
        self.pose_lost_timeout_s = 1.5
        self._start_ts = None
        self._last_pose_ts = None

    def start(self, params=None):
        params = params or {}

        self.target_yaw = float(params.get("target_yaw", 0.0))
        self.yaw_tolerance = float(params.get("yaw_tolerance", 0.05))
        self.angular_speed = abs(float(params.get("angular_speed", 0.6)))
        self.pose_lost_timeout_s = float(params.get("pose_lost_timeout_s", 1.5))

        self._start_ts = rospy.Time.now().to_sec()
        self._last_pose_ts = None

        # 旋转前先取消导航，避免 move_base 残留输出速度
        self.skill_manager.cancel_nav_goal()
        self.skill_manager.publish_stop_velocity()

        self._status = RUNNING

        rospy.loginfo(
            "[%s] RotateSkill start: target_yaw=%.3f tol=%.3f angular_speed=%.3f",
            self.skill_manager.ns,
            self.target_yaw,
            self.yaw_tolerance,
            self.angular_speed,
        )

    def update(self):
        pose = self.skill_manager.get_current_pose()
        now = rospy.Time.now().to_sec()

        if pose is None:
            if self._last_pose_ts is None:
                self._last_pose_ts = self._start_ts if self._start_ts is not None else now

            if (now - self._last_pose_ts) > self.pose_lost_timeout_s:
                rospy.logwarn(
                    "[%s] RotateSkill failed: pose lost for %.2fs",
                    self.skill_manager.ns,
                    now - self._last_pose_ts,
                )
                self.skill_manager.publish_stop_velocity()
                self._status = FAILED
                return self._status

            self._status = RUNNING
            return self._status

        self._last_pose_ts = now

        q = pose.orientation
        _, _, current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        err = self._normalize_angle(self.target_yaw - current_yaw)

        if abs(err) <= self.yaw_tolerance:
            self.skill_manager.publish_stop_velocity()
            self._status = SUCCESS
            return self._status

        cmd = Twist()
        cmd.angular.z = self.angular_speed if err > 0.0 else -self.angular_speed
        self.skill_manager.publish_cmd_vel(cmd)
        self._status = RUNNING
        return self._status

    def stop(self):
        self.skill_manager.publish_stop_velocity()

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))