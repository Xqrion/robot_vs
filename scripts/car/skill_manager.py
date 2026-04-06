#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading

import actionlib  # 预留给后续 action client 集成使用。
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseActionResult
from geometry_msgs.msg import PoseWithCovarianceStamped
from robot_vs.msg import FireEvent
from robot_vs.msg import RobotState

from skills.base_skill import RUNNING, FAILED


class SkillManager(object):
    """管理小车技能所需的 ROS 发布器与订阅器。

    职责：
    - 向 /<ns>/move_base_simple/goal 发布导航目标
    - 向 /<ns>/cmd_vel 发布速度指令
    - 订阅 /<ns>/move_base/result 跟踪导航结果
    - 提供技能对象的工厂创建方法
    """

    def __init__(self, ns):
        self.ns = str(ns)
        self._lock = threading.RLock()

        self.nav_status_code = -1  # -1 表示尚未收到导航结果
        self._latest_pose = None
        self._latest_twist = Twist()

        self.active_skill = None
        self.active_action = "NONE"

        self._feedback = {
            "task_id": 0,
            "current_action": "NONE",
            "task_status": "IDLE",
            "mode": 0,
            "reason": "",
        }

        self.team = int(rospy.get_param("~team", 0))
        self.default_hp = float(rospy.get_param("~default_hp", 100.0))
        self.default_ammo = float(rospy.get_param("~default_ammo", 50.0))

        self._goal_pub = rospy.Publisher(
            "/{}/move_base_simple/goal".format(self.ns),
            PoseStamped,
            queue_size=1,
        )
        self._cmd_vel_pub = rospy.Publisher(
            "/{}/cmd_vel".format(self.ns),
            Twist,
            queue_size=1,
        )
        self._state_pub = rospy.Publisher(
            "/{}/robot_state".format(self.ns),
            RobotState,
            queue_size=10,
        )
        self._fire_event_pub = rospy.Publisher(
            "/{}/fire_event".format(self.ns),
            FireEvent,
            queue_size=10,
        )

        self._odom_sub = rospy.Subscriber(
            "/{}/odom".format(self.ns),
            Odometry,
            self._odom_cb,
            queue_size=10,
        )
        self._amcl_sub = rospy.Subscriber(
            "/{}/amcl_pose".format(self.ns),
            PoseWithCovarianceStamped,
            self._amcl_pose_cb,
            queue_size=10,
        )
        self._nav_result_sub = rospy.Subscriber(
            "/{}/move_base/result".format(self.ns),
            MoveBaseActionResult,
            self._nav_result_cb,
            queue_size=10,
        )

        self._state_timer = rospy.Timer(rospy.Duration(0.1), self._publish_robot_state)

        rospy.loginfo("[%s] SkillManager initialised", self.ns)

    # ------------------------------------------------------------------
    # 发布器辅助方法
    # ------------------------------------------------------------------

    def publish_nav_goal(self, goal):
        """向 move_base_simple 发布 PoseStamped 目标。"""
        self._goal_pub.publish(goal)

    def publish_stop_velocity(self):
        """发布零速度 Twist，使机器人立即停止。"""
        self._cmd_vel_pub.publish(Twist())

    def publish_cmd_vel(self, cmd_vel):
        self._cmd_vel_pub.publish(cmd_vel)

    def publish_fire_event(self, x, y, yaw):
        msg = FireEvent()
        msg.shooter_ns = self.ns
        msg.x = float(x)
        msg.y = float(y)
        msg.yaw = float(yaw)
        self._fire_event_pub.publish(msg)

    # ------------------------------------------------------------------
    # 导航状态
    # ------------------------------------------------------------------

    def reset_nav_status(self):
        """发送新目标前清空上一次导航结果。"""
        self.nav_status_code = -1

    def _nav_result_cb(self, msg):
        self.nav_status_code = msg.status.status

    def _odom_cb(self, msg):
        with self._lock:
            self._latest_twist = msg.twist.twist
            if self._latest_pose is None:
                self._latest_pose = msg.pose.pose

    def _amcl_pose_cb(self, msg):
        with self._lock:
            self._latest_pose = msg.pose.pose

    # ------------------------------------------------------------------
    # 技能生命周期与工厂
    # ------------------------------------------------------------------

    def make_skill(self, action_type, task):
        """根据 *action_type* 创建对应技能实例。

        参数：
            action_type (str): 例如 "GOTO"、"STOP"
            task (dict): 来自 TaskDispatcher 的完整任务字典

        返回：
            BaseSkill 子类实例；若动作未知则回退为 StopSkill。
        """
        # 在此处导入以避免模块加载阶段出现循环依赖。
        from skills.goto_skill import GoToSkill
        from skills.stop_skill import StopSkill
        from skills.attack_skill import AttackSkill

        action = str(action_type).upper()
        if action == "GOTO":
            return GoToSkill(self)
        elif action == "STOP":
            return StopSkill(self)
        elif action == "ATTACK":
            return AttackSkill(self)
        else:
            rospy.logwarn(
                "[%s] SkillManager: unknown action_type '%s', defaulting to StopSkill",
                self.ns, action_type,
            )
            return StopSkill(self)

    def switch_skill(self, action_type, task):
        self.stop_active_skill()
        self.active_action = str(action_type).upper()
        self.active_skill = self.make_skill(action_type, task)
        try:
            self.active_skill.start(task)
        except Exception as exc:
            rospy.logwarn("[%s] skill.start failed: %s", self.ns, exc)
            self.active_skill = self.make_skill("STOP", task)
            self.active_action = "STOP"
            try:
                self.active_skill.start(task)
            except Exception as stop_exc:
                rospy.logwarn("[%s] fallback StopSkill start failed: %s", self.ns, stop_exc)

    def update_active_skill(self):
        if self.active_skill is None:
            return RUNNING
        try:
            return self.active_skill.update()
        except Exception as exc:
            rospy.logwarn("[%s] skill.update failed: %s", self.ns, exc)
            return FAILED

    def stop_active_skill(self):
        if self.active_skill is None:
            return
        try:
            self.active_skill.stop()
        except Exception as exc:
            rospy.logwarn("[%s] skill.stop failed: %s", self.ns, exc)
        self.active_skill = None

    def set_task_feedback(self, task_id, current_action, task_status, mode):
        with self._lock:
            self._feedback["task_id"] = int(task_id)
            self._feedback["current_action"] = str(current_action)
            self._feedback["task_status"] = str(task_status)
            self._feedback["mode"] = int(mode)

    def get_current_pose(self):
        with self._lock:
            return self._latest_pose

    # ------------------------------------------------------------------
    # RobotState 发布
    # ------------------------------------------------------------------

    def _publish_robot_state(self, _event):
        msg = RobotState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.robot_ns = self.ns
        msg.team = self.team
        msg.hp = self.default_hp
        msg.ammo = self.default_ammo
        msg.alive = True
        msg.in_combat = (self.active_action == "ATTACK")

        with self._lock:
            if self._latest_pose is not None:
                msg.pose = self._latest_pose
            msg.twist = self._latest_twist
            msg.current_task_id = self._feedback["task_id"]
            msg.current_action = self._feedback["current_action"]
            msg.task_status = self._feedback["task_status"]
            msg.mode = self._feedback["mode"]

        self._state_pub.publish(msg)
