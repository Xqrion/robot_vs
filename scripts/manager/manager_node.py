#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy

import rospy

from battle_state_formatter import BattleStateFormatter
from global_observer import GlobalObserver
from llm_client import LLMClient
from task_dispatcher import TaskDispatcher


class TeamManager(object):
	"""ROS1 团队管理主节点。

	核心循环：
	  1) 观测全局状态
	  2) 格式化规划输入
	  3) 调用 LLM 规划器生成任务
	  4) 分发任务
	"""

	def __init__(self, team_color="red", my_cars=None, loop_hz=0.2,
				 state_timeout_s=2.0, default_patrol_points=None,
				 enemy_topic="/referee/enemy_state",
				 llm_enabled=False, llm_api_key="", llm_model="moonshot-v1-8k", llm_timeout_s=10.0,
				 observer=None, formatter=None, llm_client=None, dispatcher=None):
		if my_cars is None:
			my_cars = []
		self.team_color = str(team_color)
		self.my_cars = list(my_cars)
		self.loop_hz = float(loop_hz)
		self.state_timeout_s = float(state_timeout_s)
		self.default_patrol_points = list(default_patrol_points) if default_patrol_points else []
		self.enemy_topic = str(enemy_topic)

		self.observer = observer if observer is not None else GlobalObserver(
			my_cars=self.my_cars,
			state_timeout=self.state_timeout_s,
			enemy_topic=self.enemy_topic,
		)
		self.formatter = formatter if formatter is not None else BattleStateFormatter()
		self.llm_client = llm_client if llm_client is not None else LLMClient(
			patrol_points=(self.default_patrol_points or None),
			enabled=llm_enabled,
			api_key=llm_api_key,
			model=llm_model,
			timeout_s=llm_timeout_s,
		)
		self.dispatcher = dispatcher if dispatcher is not None else TaskDispatcher(
			my_cars=self.my_cars,
		)

		rospy.loginfo(
			"TeamManager initialized: team_color=%s my_cars=%s loop_hz=%.3f state_timeout_s=%.2f enemy_topic=%s patrol_points=%s llm_enabled=%s llm_model=%s",
			self.team_color,
			self.my_cars,
			self.loop_hz,
			self.state_timeout_s,
			self.enemy_topic,
			self.default_patrol_points,
			llm_enabled,
			llm_model,
		)

	@classmethod
	def from_ros_params(cls):
		team_color = rospy.get_param("~team_color", "red")
		my_cars = rospy.get_param("~my_cars", [])
		loop_hz = rospy.get_param("~loop_hz", 0.2)
		state_timeout_s = rospy.get_param("~state_timeout_s", 2.0)
		default_patrol_points = rospy.get_param("~default_patrol_points", [])
		node_name = rospy.get_name().strip("/")
		default_enemy_topic = "/{}/enemy_state".format(node_name) if node_name else "/referee/enemy_state"
		enemy_topic = rospy.get_param("~enemy_topic", default_enemy_topic)

		# LLM 配置（从 YAML 中的 llm: 子键读取）
		llm_cfg = rospy.get_param("~llm", {})
		if not isinstance(llm_cfg, dict):
			llm_cfg = {}
		llm_enabled = bool(llm_cfg.get("enabled", False))
		llm_model = str(llm_cfg.get("model", "moonshot-v1-8k"))
		llm_timeout_s = float(llm_cfg.get("timeout_s", 10.0))
		llm_api_key = str(llm_cfg.get("api_key", ""))

		cls._validate_params(team_color, my_cars, loop_hz, state_timeout_s, default_patrol_points, enemy_topic)
		return cls(
			team_color=team_color,
			my_cars=my_cars,
			loop_hz=loop_hz,
			state_timeout_s=state_timeout_s,
			default_patrol_points=default_patrol_points,
			enemy_topic=enemy_topic,
			llm_enabled=llm_enabled,
			llm_api_key=llm_api_key,
			llm_model=llm_model,
			llm_timeout_s=llm_timeout_s,
		)

	@staticmethod
	def _validate_params(team_color, my_cars, loop_hz, state_timeout_s, default_patrol_points, enemy_topic):
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

		if not isinstance(enemy_topic, str) or not enemy_topic:
			raise ValueError("~enemy_topic must be a non-empty string")

	def build_fallback_tasks(self):
		fallback = {}
		for ns in self.my_cars:
			fallback[ns] = {
				"action": "STOP",
				"target": {"x": 0.0, "y": 0.0},
				"mode": 0,
				"reason": "fallback_on_exception in manager.py",
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
		# 当参数非法时使用保守默认值，确保节点保持可运行。
		node_name = rospy.get_name().strip("/")
		default_enemy_topic = "/{}/enemy_state".format(node_name) if node_name else "/referee/enemy_state"
		manager = TeamManager(team_color="red", my_cars=[], loop_hz=1, state_timeout_s=5.0, enemy_topic=default_enemy_topic)

	manager.run()


if __name__ == "__main__":
	try:
		main()
	except rospy.ROSInterruptException:
		pass
