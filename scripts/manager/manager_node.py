#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import rospy

from battle_state_formatter import BattleStateFormatter
from global_observer import GlobalObserver
from llm_client import LLMClient
from task_dispatcher import TaskDispatcher


class TeamManager(object):
	"""ROS1 team manager main node.

	Core loop:
	  1) observe global state
	  2) format planner input
	  3) ask LLM planner for tasks
	  4) dispatch tasks
	"""

	def __init__(self, team_color="red", my_cars=None, loop_hz=0.2,
				 state_timeout_s=2.0, default_patrol_points=None,
				 observer=None, formatter=None, llm_client=None, dispatcher=None):
		if my_cars is None:
			my_cars = []
		self.team_color = str(team_color)
		self.my_cars = list(my_cars)
		self.loop_hz = float(loop_hz)
		self.state_timeout_s = float(state_timeout_s)
		self.default_patrol_points = list(default_patrol_points) if default_patrol_points else []

		self.observer = observer if observer is not None else GlobalObserver(
			my_cars=self.my_cars,
			state_timeout=self.state_timeout_s,
		)
		self.formatter = formatter if formatter is not None else BattleStateFormatter()
		self.llm_client = llm_client if llm_client is not None else LLMClient(
			patrol_points=(self.default_patrol_points or None),
		)
		self.dispatcher = dispatcher if dispatcher is not None else TaskDispatcher(
			my_cars=self.my_cars,
		)

		rospy.loginfo(
			"TeamManager initialized: team_color=%s my_cars=%s loop_hz=%.3f state_timeout_s=%.2f patrol_points=%s",
			self.team_color,
			self.my_cars,
			self.loop_hz,
			self.state_timeout_s,
			self.default_patrol_points,
		)

	@classmethod
	def from_ros_params(cls):
		team_color = rospy.get_param("~team_color", "red")
		my_cars = rospy.get_param("~my_cars", [])
		loop_hz = rospy.get_param("~loop_hz", 0.2)
		state_timeout_s = rospy.get_param("~state_timeout_s", 2.0)
		default_patrol_points = rospy.get_param("~default_patrol_points", [])

		cls._validate_params(team_color, my_cars, loop_hz, state_timeout_s, default_patrol_points)
		return cls(
			team_color=team_color,
			my_cars=my_cars,
			loop_hz=loop_hz,
			state_timeout_s=state_timeout_s,
			default_patrol_points=default_patrol_points,
		)

	@staticmethod
	def _validate_params(team_color, my_cars, loop_hz, state_timeout_s, default_patrol_points):
		if not isinstance(team_color, str):
			raise ValueError("~team_color must be a string")

		if not isinstance(my_cars, list):
			raise ValueError("~my_cars must be a list of strings")

		if not all(isinstance(car, str) and car for car in my_cars):
			raise ValueError("~my_cars must contain non-empty strings only")

		try:
			hz = float(loop_hz)
		except Exception:
			raise ValueError("~loop_hz must be a float")

		if hz <= 0.0:
			raise ValueError("~loop_hz must be > 0")

		try:
			timeout_s = float(state_timeout_s)
		except Exception:
			raise ValueError("~state_timeout_s must be a float")

		if timeout_s <= 0.0:
			raise ValueError("~state_timeout_s must be > 0")

		if not isinstance(default_patrol_points, list):
			raise ValueError("~default_patrol_points must be a list")

	def build_fallback_tasks(self):
		fallback = {}
		for ns in self.my_cars:
			fallback[ns] = {
				"action": "STOP",
				"target": {"x": 0.0, "y": 0.0},
				"mode": 0,
				"reason": "fallback_on_exception",
				"timeout": 2.0,
			}
		return fallback

	def run_cycle(self):
		state = self.observer.get_battle_state()#状态字典
		prompt_input = self.formatter.build(state, self.team_color, self.my_cars)
		tasks = self.llm_client.plan_tasks(prompt_input)#任务字典
		self.dispatcher.dispatch(tasks)
		return tasks

	def run(self):
		rate = rospy.Rate(self.loop_hz)
		while not rospy.is_shutdown():
			try:
				self.run_cycle()
			except Exception as exc:
				rospy.logwarn("TeamManager cycle failed: %s", exc)
				fallback_tasks = self.build_fallback_tasks()
				try:
					self.dispatcher.dispatch(copy.deepcopy(fallback_tasks))
				except Exception as dispatch_exc:
					rospy.logwarn("Fallback dispatch failed: %s", dispatch_exc)
			rate.sleep()


def main():
	rospy.init_node("team_manager")

	try:
		manager = TeamManager.from_ros_params()
	except Exception as exc:
		rospy.logwarn("TeamManager param/init error: %s", exc)
		# Keep node alive with conservative defaults when params are invalid.
		manager = TeamManager(team_color="red", my_cars=[], loop_hz=0.2, state_timeout_s=2.0)

	manager.run()


if __name__ == "__main__":
	try:
		main()
	except rospy.ROSInterruptException:
		pass
