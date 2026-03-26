#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading

import rospy
from robot_vs.msg import RobotState, TaskCommand
from std_msgs.msg import Header

from skills.base_skill import RUNNING, SUCCESS, FAILED


class TaskEngine(object):
    """Receive TaskCommand messages and drive skill execution.

    Responsibilities:
    1) Subscribe to /<ns>/task_cmd (TaskCommand) for incoming tasks.
    2) Instantiate the appropriate skill via SkillManager on task change.
    3) On every tick(), call the current skill's update() and handle
       transitions (RUNNING → SUCCESS / FAILED).
    4) Publish /<ns>/robot_state (RobotState) with current execution
       feedback so the Manager can monitor progress.
    """

    def __init__(self, ns, skill_manager):
        self.ns = str(ns)
        self.skill_manager = skill_manager

        self._lock = threading.RLock()
        self._current_task = None   # latest TaskCommand msg
        self._current_skill = None  # active BaseSkill instance
        self._task_status = "IDLE"  # IDLE / RUNNING / SUCCESS / FAILED
        self._current_action = "NONE"

        self._task_sub = rospy.Subscriber(
            "/{}/task_cmd".format(self.ns),
            TaskCommand,
            self._task_cmd_cb,
            queue_size=5,
        )
        self._state_pub = rospy.Publisher(
            "/{}/robot_state".format(self.ns),
            RobotState,
            queue_size=5,
        )

        rospy.loginfo("[%s] TaskEngine initialised", self.ns)

    # ------------------------------------------------------------------
    # Subscriber callback
    # ------------------------------------------------------------------

    def _task_cmd_cb(self, msg):
        with self._lock:
            current = self._current_task
            if current is not None and current.task_id == msg.task_id:
                return  # same task, ignore duplicate

            rospy.loginfo(
                "[%s] TaskEngine: new task task_id=%d action=%s target=(%.2f, %.2f)",
                self.ns, msg.task_id, msg.action_type, msg.target_x, msg.target_y,
            )

            # Stop the old skill
            if self._current_skill is not None:
                try:
                    self._current_skill.stop()
                except Exception as exc:
                    rospy.logwarn("[%s] skill.stop() raised: %s", self.ns, exc)

            # Build task dict for the skill factory
            task_dict = {
                "task_id": msg.task_id,
                "action_type": msg.action_type,
                "target_x": msg.target_x,
                "target_y": msg.target_y,
                "mode": msg.mode,
                "reason": msg.reason,
                "timeout": msg.timeout,
            }

            new_skill = self.skill_manager.make_skill(msg.action_type, task_dict)
            try:
                new_skill.start()
            except Exception as exc:
                rospy.logwarn("[%s] skill.start() raised: %s", self.ns, exc)

            self._current_task = msg
            self._current_skill = new_skill
            self._task_status = RUNNING
            self._current_action = str(msg.action_type).upper()

    # ------------------------------------------------------------------
    # Main loop step
    # ------------------------------------------------------------------

    def tick(self):
        """Called every loop iteration from car_node."""
        with self._lock:
            task = self._current_task
            skill = self._current_skill

        if skill is None:
            self._publish_robot_state()
            return

        try:
            result = skill.update()
        except Exception as exc:
            rospy.logwarn("[%s] skill.update() raised: %s", self.ns, exc)
            result = FAILED

        with self._lock:
            self._task_status = result

        self._publish_robot_state()

    # ------------------------------------------------------------------
    # State publisher
    # ------------------------------------------------------------------

    def _publish_robot_state(self):
        with self._lock:
            task = self._current_task
            task_status = self._task_status
            current_action = self._current_action

        msg = RobotState()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.robot_ns = self.ns
        msg.alive = True
        msg.current_task_id = int(task.task_id) if task is not None else 0
        msg.current_action = current_action
        msg.task_status = task_status

        self._state_pub.publish(msg)
