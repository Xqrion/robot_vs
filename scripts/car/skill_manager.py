#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import PoseStamped, Twist
from move_base_msgs.msg import MoveBaseActionResult


class SkillManager(object):
    """Manage ROS publishers and subscribers needed by car skills.

    Responsibilities:
    - Publish navigation goals to /<ns>/move_base_simple/goal
    - Publish velocity commands to /<ns>/cmd_vel
    - Subscribe to /<ns>/move_base/result to track navigation outcome
    - Provide factory methods to instantiate skill objects
    """

    def __init__(self, ns):
        self.ns = str(ns)
        self.nav_status_code = -1  # -1 = no result yet

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
        self._nav_result_sub = rospy.Subscriber(
            "/{}/move_base/result".format(self.ns),
            MoveBaseActionResult,
            self._nav_result_cb,
            queue_size=10,
        )

        rospy.loginfo("[%s] SkillManager initialised", self.ns)

    # ------------------------------------------------------------------
    # Publisher helpers
    # ------------------------------------------------------------------

    def publish_nav_goal(self, goal):
        """Send a PoseStamped goal to move_base_simple."""
        self._goal_pub.publish(goal)

    def publish_stop_velocity(self):
        """Publish zero Twist to stop the robot immediately."""
        self._cmd_vel_pub.publish(Twist())

    # ------------------------------------------------------------------
    # Navigation status
    # ------------------------------------------------------------------

    def reset_nav_status(self):
        """Clear the last navigation result before sending a new goal."""
        self.nav_status_code = -1

    def _nav_result_cb(self, msg):
        self.nav_status_code = msg.status.status

    # ------------------------------------------------------------------
    # Skill factory
    # ------------------------------------------------------------------

    def make_skill(self, action_type, task):
        """Instantiate the appropriate skill for *action_type*.

        Args:
            action_type (str): e.g. "GOTO", "STOP"
            task (dict): full task dict from TaskDispatcher

        Returns:
            BaseSkill subclass instance, or None if unknown action.
        """
        # Import here to avoid circular imports at module load time.
        from skills.goto_skill import GoToSkill
        from skills.stop_skill import StopSkill

        action = str(action_type).upper()
        if action == "GOTO":
            return GoToSkill(
                self,
                target_x=task.get("target_x", 0.0),
                target_y=task.get("target_y", 0.0),
            )
        elif action == "STOP":
            return StopSkill(self)
        else:
            rospy.logwarn(
                "[%s] SkillManager: unknown action_type '%s', defaulting to StopSkill",
                self.ns, action_type,
            )
            return StopSkill(self)
