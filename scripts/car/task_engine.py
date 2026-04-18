#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading

import rospy
from robot_vs.msg import TaskCommand

from skills.base_skill import RUNNING, SUCCESS, FAILED
from skill_manager import SkillManager


class TaskEngine(object):
    """维护当前任务并驱动技能执行。

     职责：
     1) 保存 car_node 下发的当前任务快照。
     2) 任务切换时通过 SkillManager 创建并切换对应技能。
     3) 每次 tick() 调用当前技能 update()，处理
         RUNNING -> SUCCESS / FAILED 的状态转移。
     4) 监控任务超时并执行 STOP 兜底。
    """

    def __init__(self, ns, skill_manager):
        self.ns = str(ns)
        self.skill_manager = skill_manager

        self._lock = threading.RLock()
        self._current_task = None   # 最近一次任务字典
        self._task_status = "IDLE"  # IDLE / RUNNING / SUCCESS / FAILED
        self._current_action = "NONE"
        self._task_start_t = None

        rospy.loginfo("[%s] TaskEngine initialised", self.ns)

    # ------------------------------------------------------------------
    # 任务接收
    # ------------------------------------------------------------------

    '''
    每当car_node接收到新的task时，都会调用accept_task()方法
    启动新技能,然后把任务状态置为 RUNNING
    '''
    def accept_task(self, msg):
        if not isinstance(msg, TaskCommand):
            raise ValueError("accept_task expects TaskCommand")

        with self._lock:
            current = self._current_task
            if current is not None and int(current.get("task_id", 0)) == int(msg.task_id):
                return  # 相同 task_id 视为重复任务，直接忽略

            rospy.loginfo(
                "[%s] TaskEngine: new task task_id=%d action=%s target=(%.2f, %.2f, yaw=%.2f) ",
                self.ns, msg.task_id, msg.action_type, msg.target_x, msg.target_y, msg.target_yaw
            )

            task_dict = {
                "task_id": msg.task_id,
                "action_type": msg.action_type,
                "target_x": msg.target_x,
                "target_y": msg.target_y,
                "target_yaw": msg.target_yaw,
                "mode": msg.mode,
                "reason": msg.reason,
                "timeout": msg.timeout,
            }
            try:
                self.skill_manager.switch_skill(msg.action_type, task_dict)
            except Exception as exc:
                rospy.logwarn("[%s] switch_skill failed: %s", self.ns, exc)

            self._current_task = task_dict
            self._task_start_t = rospy.Time.now().to_sec()
            self._task_status = RUNNING
            self._current_action = str(msg.action_type).upper()
            self.skill_manager.set_task_feedback(
                task_id=int(msg.task_id),
                current_action=self._current_action,
                task_status=self._task_status,
                mode=int(msg.mode),
            )

    # ------------------------------------------------------------------
    # 主循环步进
    # ------------------------------------------------------------------

    '''
    每个一个周期查看小车task的运行状态
    1. 如果当前没有task，则发送IDLE状态反馈
    2. 如果当前task正在运行，检查是否超时，如果超时则切换到STOP技能，并发送FAILED状态反馈
    3. 否则调用当前技能的update()方法，并根据返回结果更新任务状态
    '''
    def tick(self):
        """由 car_node 在每次循环中调用。"""
        with self._lock:
            task = self._current_task
            task_status = self._task_status
            task_start_t = self._task_start_t

        if task is None:
            self.skill_manager.set_task_feedback(
                task_id=0,
                current_action="NONE",
                task_status="IDLE",
                mode=0,
            )
            return

        if task_status == RUNNING and self._is_task_timeout(task, task_start_t):
            rospy.logwarn("[%s] task timeout: task_id=%s", self.ns, task.get("task_id"))
            with self._lock:
                self._task_status = FAILED
            self.skill_manager.stop_active_skill()
            try:
                self.skill_manager.switch_skill("STOP", self._build_timeout_stop_task(task))
            except Exception as exc:
                rospy.logwarn("[%s] timeout stop switch failed: %s", self.ns, exc)
            self.skill_manager.set_task_feedback(
                task_id=int(task.get("task_id", 0)),
                current_action=str(task.get("action_type", "NONE")).upper(),
                task_status=FAILED,
                mode=int(task.get("mode", 0)),
            )
            return

        try:
            result = self.skill_manager.update_active_skill()
        except Exception as exc:
            rospy.logwarn("[%s] skill.update() raised: %s", self.ns, exc)
            result = FAILED

        with self._lock:
            self._task_status = result

        self.skill_manager.set_task_feedback(
            task_id=int(task.get("task_id", 0)),
            current_action=self._current_action,
            task_status=self._task_status,
            mode=int(task.get("mode", 0)),
        )

    def _is_task_timeout(self, task, task_start_t):
        if task_start_t is None:
            return False

        timeout = float(task.get("timeout", 0.0) or 0.0)
        if timeout <= 0.0:
            return False

        elapsed = rospy.Time.now().to_sec() - float(task_start_t)
        return elapsed > timeout

    def _build_timeout_stop_task(self, task):
        return {
            "task_id": int(task.get("task_id", 0)),
            "action_type": "STOP",
            "target_x": 0.0,
            "target_y": 0.0,
            "target_yaw": 0.0,
            "mode": 0,
            "reason": "timeout",
            "timeout": 0.0,
        }
