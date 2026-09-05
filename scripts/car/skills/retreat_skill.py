#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import Twist

from skills.base_skill import BaseSkill, RUNNING, SUCCESS, FAILED


class RetreatSkill(BaseSkill):
    """
    撤退技能：向远离最近敌人的方向移动，保命优先。
    """

    def __init__(self, skill_manager):
        super(RetreatSkill, self).__init__(skill_manager)
        self._target_x = 0.0
        self._target_y = 0.0
        self._start_time = None
        self._timeout = 6.0
        self._retreat_speed = 0.4
        self._arrival_tolerance = 0.5

    def start(self, task):
        """初始化撤退目标"""
        target = task.get('target', {})
        self._target_x = float(target.get('x', 0.0))
        self._target_y = float(target.get('y', 0.0))
        self._timeout = float(task.get('timeout', 6.0))
        self._start_time = rospy.Time.now().to_sec()

        rospy.loginfo("[RetreatSkill] start: target=(%.2f, %.2f)", self._target_x, self._target_y)

        # 取消任何正在进行的导航
        self._skill_manager.publish_nav_cancel()
        self._skill_manager.reset_nav_status()

    def update(self):
        """每帧执行：向撤退目标移动"""
        # 超时检查
        if rospy.Time.now().to_sec() - self._start_time > self._timeout:
            rospy.logwarn("[RetreatSkill] timeout")
            self._skill_manager.publish_stop_velocity()
            return FAILED

        # 获取当前位置
        pose = self._skill_manager.get_current_pose()
        if pose is None:
            rospy.logwarn("[RetreatSkill] no pose, waiting...")
            return RUNNING

        dx = self._target_x - pose.position.x
        dy = self._target_y - pose.position.y
        distance = math.hypot(dx, dy)

        # 到达目标点
        if distance < self._arrival_tolerance:
            rospy.loginfo("[RetreatSkill] arrived at retreat point")
            self._skill_manager.publish_stop_velocity()
            return SUCCESS

        # 计算目标方向角
        angle_to_target = math.atan2(dy, dx)
        current_yaw = self._skill_manager.get_current_yaw()
        if current_yaw is None:
            return RUNNING

        # 角度差
        angle_diff = angle_to_target - current_yaw
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi

        # 发布速度指令
        cmd = Twist()
        if abs(angle_diff) > 0.15:
            # 先转向目标方向
            cmd.linear.x = 0.0
            cmd.angular.z = 0.8 * angle_diff
        else:
            # 全速前进
            cmd.linear.x = self._retreat_speed
            cmd.angular.z = 0.0

        self._skill_manager.publish_cmd_vel(cmd)
        return RUNNING

    def stop(self):
        """停止撤退"""
        rospy.loginfo("[RetreatSkill] stop")
        self._skill_manager.publish_stop_velocity()
        self._skill_manager.publish_nav_cancel()
