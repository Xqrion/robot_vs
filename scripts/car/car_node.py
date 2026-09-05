#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

import rospy

from robot_vs.msg import TaskCommand

# 允许导入同级包（skill_manager、task_engine、skills/）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_manager import SkillManager
from task_engine import TaskEngine


class CarAgent(object):
    """单个小车智能体主入口。

    身份仅由启动时的 ROS 命名空间决定
    （例如 /robot_red_1），因此同一份程序可服务双方全部小车。

    参数（从 ROS 参数服务器读取）：
        ~loop_hz (float, 默认 10.0) - 主循环频率（Hz）
    """

    def __init__(self, ns=None, loop_hz=None, skill_manager=None, task_engine=None):
        # 解析命名空间：优先使用显式参数，其次使用 ROS 命名空间。
        if ns is None:
            ns = rospy.get_namespace().strip("/")  # 命名空间来自 launch 的 <group ns=...> 配置。
        self.ns = str(ns) if ns else "car"

        if loop_hz is None:
            loop_hz = rospy.get_param("~loop_hz", 10.0)
        self.loop_hz = float(loop_hz)

        self.skill_manager = skill_manager if skill_manager is not None else SkillManager(self.ns)
        self.task_engine = task_engine if task_engine is not None else TaskEngine(self.ns, self.skill_manager)

        self._task_sub = rospy.Subscriber(
             "/{}/car_task".format(self.ns),
    	     TaskCommand,
             self._task_cmd_cb,
             queue_size=10,
        )

        rospy.loginfo(
            "CarAgent initialised: ns=%s loop_hz=%.1f",
            self.ns, self.loop_hz,
        )

    def run(self):
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            try:
                self.task_engine.tick()
            except Exception as exc:
                rospy.logwarn("CarAgent tick failed: %s", exc)
            rate.sleep()

    def _task_cmd_cb(self, msg):
        try:
            self.task_engine.accept_task(msg)
        except Exception as exc:
            rospy.logwarn("[%s] task callback failed: %s", self.ns, exc)


def main():
    rospy.init_node("car_agent_node")

    try:
        agent = CarAgent()
    except Exception as exc:
        rospy.logerr("CarAgent init failed: %s", exc)
        return

    agent.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
