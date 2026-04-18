#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import rospy
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

from skills.base_skill import BaseSkill, RUNNING, FAILED


class AttackSkill(BaseSkill):
    """朝向目标后触发开火事件，并持续处于攻击状态。"""

    def __init__(self, skill_manager):
        super(AttackSkill, self).__init__(skill_manager)
        self.target_x = 0.0
        self.target_y = 0.0
        self.yaw_tolerance = 0.02
        self.angular_speed = 0.7
        self.fire_cooldown_s = 2.0
        self.pose_lost_timeout_s = 1.5
        self._last_fire_ts = None
        self._start_ts = None
        self._last_pose_ts = None

    def start(self, params=None):
        params = params or {}
        self.target_x = float(params.get("target_x", 0.0))
        self.target_y = float(params.get("target_y", 0.0))
        self.yaw_tolerance = float(params.get("yaw_tolerance", 0.02))
        self.angular_speed = abs(float(params.get("angular_speed", 0.7)))
        self.fire_cooldown_s = float(params.get("fire_cooldown_s", 2.0))
        self.pose_lost_timeout_s = float(params.get("pose_lost_timeout_s", 1.5))
        self._last_fire_ts = None
        self._start_ts = rospy.Time.now().to_sec()
        self._last_pose_ts = None
        self._status = RUNNING

        rospy.loginfo(
            "[%s] AttackSkill start: target=(%.2f, %.2f) yaw_tol=%.3f cooldown=%.2fs",
            self.skill_manager.ns,
            self.target_x,
            self.target_y,
            self.yaw_tolerance,
            self.fire_cooldown_s,
        )

    def update(self):
        pose = self.skill_manager.get_current_pose()
        now = rospy.Time.now().to_sec()

        if pose is None:
            if self._last_pose_ts is None:
                self._last_pose_ts = self._start_ts if self._start_ts is not None else now
            if (now - self._last_pose_ts) > self.pose_lost_timeout_s:
                rospy.logwarn(
                    "[%s] AttackSkill failed: pose lost for %.2fs",
                    self.skill_manager.ns,
                    now - self._last_pose_ts,
                )
                self.skill_manager.publish_stop_velocity()
                self._status = FAILED
                return self._status

            self._status = RUNNING
            return self._status

        self._last_pose_ts = now

        cmd = Twist()
        q = pose.orientation
        _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        desired_yaw = math.atan2(self.target_y - pose.position.y, self.target_x - pose.position.x)
        err = self._normalize_angle(desired_yaw - yaw)

        if abs(err) > self.yaw_tolerance:
            cmd.angular.z = self.angular_speed if err > 0.0 else -self.angular_speed
            self.skill_manager.publish_cmd_vel(cmd)
            self._status = RUNNING
            return self._status

        self.skill_manager.publish_stop_velocity()
        can_fire = (
            self._last_fire_ts is None or
            (now - self._last_fire_ts) >= self.fire_cooldown_s
        )
        if can_fire:
            self.skill_manager.publish_fire_event(
                x=pose.position.x,
                y=pose.position.y,
                yaw=yaw,
            )
            self._last_fire_ts = now
            rospy.loginfo(
                "[%s] AttackSkill fire_event published: pose=(%.2f, %.2f) yaw=%.2f",
                self.skill_manager.ns,
                pose.position.x,
                pose.position.y,
                yaw,
            )

        self._status = RUNNING
        return self._status

    def stop(self):
        self.skill_manager.publish_stop_velocity()

    @staticmethod
    def _normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))
