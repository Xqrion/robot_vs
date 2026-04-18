#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import euler_from_quaternion

from skills.base_skill import BaseSkill, RUNNING, SUCCESS, FAILED


class GoToSkill(BaseSkill):
    """通过 move_base_simple/goal 导航到指定 (x, y) 目标。

    当 move_base 返回 SUCCEEDED（status == 3）时结束为 SUCCESS。
    当 move_base 返回失败状态时，尝试调整目标点并重试，最多重试3次。
    """

    # move_base 的 GoalStatus 状态码
    _STATUS_SUCCEEDED = 3
    _STATUS_FAILED = {4, 5, 8, 9}  # ABORTED、REJECTED、LOST 等

    def __init__(self, skill_manager, frame_id="map"):
        super(GoToSkill, self).__init__(skill_manager)
        self.target_x = 0.0
        self.target_y = 0.0
        self.frame_id = str(frame_id)
        self.retry_count = 0
        self.max_retries = 3

    def start(self, params=None):
        params = params or {}
        self.target_x = float(params.get("target_x", 0.0))
        self.target_y = float(params.get("target_y", 0.0))
        self.retry_count = 0

        # 调整目标点以避免不合理位置
        self.target_x, self.target_y = self._adjust_target(self.target_x, self.target_y)

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
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                # 调整目标点
                self.target_x, self.target_y = self._adjust_target(self.target_x, self.target_y)
                # 重新发布目标
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
                    "[%s] GoToSkill retry %d: adjusted target=(%.2f, %.2f)",
                    self.skill_manager.ns, self.retry_count, self.target_x, self.target_y,
                )
            else:
                rospy.logwarn(
                    "[%s] GoToSkill failed after %d retries: move_base status=%d",
                    self.skill_manager.ns, self.max_retries, nav_status,
                )
                self._status = FAILED
        else:
            self._status = RUNNING

        return self._status

    def _adjust_target(self, target_x, target_y):
        """调整目标点以避免不合理位置，如在障碍物上、机器人当前位置或地图外。"""
        # 首先检查地图边界
        map_info = self.skill_manager.get_map_info()
        if map_info:
            min_x = map_info['origin_x']
            max_x = map_info['origin_x'] + map_info['width'] * map_info['resolution']
            min_y = map_info['origin_y']
            max_y = map_info['origin_y'] + map_info['height'] * map_info['resolution']
            target_x = max(min_x, min(target_x, max_x))
            target_y = max(min_y, min(target_y, max_y))

        pose = self.skill_manager.get_current_pose()
        if pose is None:
            return target_x, target_y

        rx = pose.position.x
        ry = pose.position.y
        dx = target_x - rx
        dy = target_y - ry
        dist = math.sqrt(dx**2 + dy**2)
        offset = 0.5  # 偏移距离，单位米

        if dist < 0.1:  # 目标点太靠近机器人当前位置，偏移到机器人前方
            q = pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            new_x = rx + offset * math.cos(yaw)
            new_y = ry + offset * math.sin(yaw)
        else:
            # 沿着目标方向偏移一定距离
            norm_dx = dx / dist
            norm_dy = dy / dist
            new_x = target_x + norm_dx * offset
            new_y = target_y + norm_dy * offset

        return new_x, new_y

    def stop(self):
        # 通过发布零速度进行取消；move_base 将进入空闲。
        self.skill_manager.publish_stop_velocity()
