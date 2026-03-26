#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os

import rospy

# Allow sibling-package imports (skill_manager, task_engine, skills/)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skill_manager import SkillManager
from task_engine import TaskEngine


class CarAgent(object):
    """Main entry point for a single car agent.

    Identity is determined solely by the ROS namespace it is launched
    under (e.g. /robot_red_1), so one binary serves all cars on both
    teams.

    Parameters (read from ROS param server):
        ~loop_hz (float, default 10.0) – main loop frequency in Hz
    """

    def __init__(self, ns=None, loop_hz=None, skill_manager=None, task_engine=None):
        # Resolve namespace: prefer explicit arg, then ROS namespace.
        if ns is None:
            ns = rospy.get_namespace().strip("/")
        self.ns = str(ns) if ns else "car"

        if loop_hz is None:
            loop_hz = rospy.get_param("~loop_hz", 10.0)
        self.loop_hz = float(loop_hz)

        self.skill_manager = skill_manager if skill_manager is not None else SkillManager(self.ns)
        self.task_engine = task_engine if task_engine is not None else TaskEngine(self.ns, self.skill_manager)

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
