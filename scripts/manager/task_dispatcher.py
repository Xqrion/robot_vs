#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from robot_vs.msg import TaskCommand


try:
	text_type = unicode  # type: ignore[name-defined]
	binary_type = str
except NameError:
	text_type = str
	binary_type = bytes


class TaskDispatcher(object):
	"""将团队任务发布到 /<ns>/task_cmd 话题。"""

	def __init__(self, my_cars=None, default_timeout=2.0):
		if my_cars is None:
			my_cars = rospy.get_param("~my_cars", [])
		if not isinstance(my_cars, list):
			my_cars = []

		self.my_cars = list(my_cars)
		self.default_timeout = float(default_timeout)
		self._publishers = {}
		self._task_seq = 0
		self._last_task_signature = {}
		self._last_task_id = {}

		for ns in self.my_cars:
			self._ensure_publisher(ns)

	def _ensure_publisher(self, ns):
		if ns in self._publishers:
			return self._publishers[ns]

		topic = "/{}/task_cmd".format(ns)
		self._publishers[ns] = rospy.Publisher(topic, TaskCommand, queue_size=10)
		rospy.loginfo("TaskDispatcher publisher created: %s", topic)
		return self._publishers[ns]

	def _normalize_tasks(self, tasks):
		if tasks is None:
			raise ValueError("tasks must not be None")

		if isinstance(tasks, dict):
			return tasks

		# 兼容旧版 list 格式：
		# [{"car":"robot_x","type":"idle","reason":"..."}, ...]
		rospy.logwarn("tasks is a list, using legacy format normalization. Consider updating LLM output to dict format.")
		if isinstance(tasks, list):
			normalized = {}
			for item in tasks:
				if not isinstance(item, dict):
					continue
				ns = item.get("car")
				if not ns:
					continue
				normalized[ns] = {
					"action": self._to_text(item.get("type", "STOP"), "STOP").upper(),
					"target": {
						"x": float(item.get("target_x", 0.0)),
						"y": float(item.get("target_y", 0.0)),
						"yaw": float(item.get("target_yaw", 0.0)),
					},
					"mode": int(item.get("mode", 0)),
					"reason": self._to_text(item.get("reason", "legacy format"), "legacy format"),
					"timeout": float(item.get("timeout", self.default_timeout)),
				}
			return normalized

		raise ValueError("tasks must be dict or list")

	def _next_task_id(self):
		self._task_seq += 1
		return self._task_seq

	def _safe_stop_task(self, reason):
		return {
			"action": "STOP",
			"target": {"x": 0.0, "y": 0.0,"yaw": 0.0},
			"mode": 0,
			"reason": reason,
			"timeout": self.default_timeout,
		}

	def _task_signature(self, task):
		action = self._to_text(task.get("action", "STOP"), "STOP")
		target = task.get("target", {})
		if not isinstance(target, dict):
			target = {}

		target_x = float(target.get("x", 0.0))
		target_y = float(target.get("y", 0.0))
		target_yaw = float(target.get("yaw", 0.0))
		mode = int(task.get("mode", 0))
		timeout = float(task.get("timeout", self.default_timeout))

		return (action, target_x, target_y, target_yaw, mode, timeout)

	def _to_text(self, value, default=u""):
		if value is None:
			value = default

		try:
			if isinstance(value, text_type):
				return value
			if isinstance(value, binary_type):
				return value.decode("utf-8", "replace")
			return text_type(value)
		except Exception:
			try:
				return text_type(default)
			except Exception:
				return u""

	def _assign_task_id(self, ns, task):
		signature = self._task_signature(task)
		last_signature = self._last_task_signature.get(ns)

		if last_signature == signature and ns in self._last_task_id:
			return self._last_task_id[ns]

		if "task_id" in task:
			task_id = int(task.get("task_id"))
		else:
			task_id = self._next_task_id()
		self._last_task_signature[ns] = signature
		self._last_task_id[ns] = task_id
		return task_id

	def _build_task_msg(self, ns, task):
		msg = TaskCommand()
		msg.task_id = int(self._assign_task_id(ns, task))
		msg.action_type = self._to_text(task.get("action", "STOP"), "STOP")

		target = task.get("target", {})
		if not isinstance(target, dict):
			target = {}
		msg.target_x = float(target.get("x", 0.0))
		msg.target_y = float(target.get("y", 0.0))
		msg.target_yaw = float(target.get("yaw", 0.0))

		msg.mode = int(task.get("mode", 0))
		msg.reason = self._to_text(task.get("reason", ""), "")
		msg.timeout = float(task.get("timeout", self.default_timeout))
		return msg

	def dispatch(self, tasks):
		normalized = self._normalize_tasks(tasks)

		if self.my_cars:
			target_ns_list = list(self.my_cars)
		else:
			target_ns_list = sorted(normalized.keys())

		for ns in target_ns_list:
			try:
				task = normalized.get(ns)
				if task is None:
					task = self._safe_stop_task("missing task for robot")

				pub = self._ensure_publisher(ns)
				msg = self._build_task_msg(ns, task)
				pub.publish(msg)

				rospy.loginfo(
					"dispatch ns=%s task_id=%d action=%s target=(%.2f, %.2f, yaw=%.2f) reason=%r",
					ns,
					msg.task_id,
					msg.action_type,
					msg.target_x,
					msg.target_y,
					msg.target_yaw,
					msg.reason,
				)
			except Exception as exc:
				rospy.logwarn("dispatch failed for %s: %s", ns, exc)
