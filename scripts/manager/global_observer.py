#!/usr/bin/env python
# -*- coding: utf-8 -*-

import threading

import rospy
from robot_vs.msg import RobotState


class GlobalObserver(object):
	"""Observe team robot states and referee enemy state.

	Responsibilities:
	1) subscribe /<ns>/robot_state (RobotState)
	2) subscribe /referee/enemy_state (reserved interface)
	3) cache latest state + receive timestamp
	4) provide structured battle state with stale flags
	"""

	def __init__(self, my_cars, state_timeout=2.0,
				 enemy_topic="/referee/enemy_state", enemy_timeout=None):
		if my_cars is None or not isinstance(my_cars, list):
			my_cars = []
		self.my_cars = list(my_cars)
		self.state_timeout = float(state_timeout)
		self.enemy_timeout = float(enemy_timeout) if enemy_timeout is not None else float(state_timeout)
		self.enemy_topic = enemy_topic

		self._lock = threading.RLock()

		# dict[ns] = {"state": RobotState or None, "stamp": float or None}
		self.robot_states = {}
		# dict with same shape as robot state cache
		self.enemy_state = {"state": None, "stamp": None}

		self._robot_subscribers = []

		for ns in self.my_cars:
			self.robot_states[ns] = {"state": None, "stamp": None}
			topic = "/{}/robot_state".format(ns)
			sub = rospy.Subscriber(topic, RobotState, self._robot_state_cb, callback_args=ns, queue_size=10)
			self._robot_subscribers.append(sub)

		# Enemy topic interface is intentionally generic for now.
		self._enemy_subscriber = rospy.Subscriber(
			self.enemy_topic,
			rospy.AnyMsg,
			self._enemy_state_cb,
			queue_size=10,
		)

	def _robot_state_cb(self, msg, ns):
		stamp = rospy.Time.now().to_sec()
		if hasattr(msg, "header") and hasattr(msg.header, "stamp") and msg.header.stamp:
			if msg.header.stamp.to_sec() > 0.0:
				stamp = msg.header.stamp.to_sec()
		with self._lock:
			self.robot_states[ns] = {
				"state": msg,
				"stamp": stamp,
			}

	def _enemy_state_cb(self, msg):
		now = rospy.Time.now().to_sec()
		with self._lock:
			self.enemy_state = {
				"state": msg,
				"stamp": now,
			}

	def _is_stale(self, stamp, timeout):
		if stamp is None:
			return True
		age = rospy.Time.now().to_sec() - float(stamp)
		return age > timeout

	def _msg_to_dict(self, msg):
		if msg is None:
			return None

		# rospy.AnyMsg has only raw bytes; keep a minimal structured view.
		if isinstance(msg, rospy.AnyMsg):
			return {
				"_type": "rospy.AnyMsg",
				"size": len(msg._buff) if msg._buff is not None else 0,
			}

		result = {"_type": getattr(msg, "_type", msg.__class__.__name__)}
		slots = getattr(msg, "__slots__", [])
		for key in slots:
			try:
				result[key] = getattr(msg, key)
			except Exception:
				result[key] = None
		return result

	def get_battle_state(self):
		now = rospy.Time.now().to_sec()

		with self._lock:
			friendly = {}
			for ns in self.my_cars:
				record = self.robot_states.get(ns, {"state": None, "stamp": None})
				stamp = record.get("stamp")
				friendly[ns] = {
					"stamp": stamp,
					"state": self._msg_to_dict(record.get("state")),
					"stale": self._is_stale(stamp, self.state_timeout),
				}

			enemy_stamp = self.enemy_state.get("stamp")
			enemy = {
				"topic": self.enemy_topic,
				"stamp": enemy_stamp,
				"state": self._msg_to_dict(self.enemy_state.get("state")),
				"stale": self._is_stale(enemy_stamp, self.enemy_timeout),
			}

		return {
			"stamp": now,
			"friendly": friendly,
			"enemy": enemy,
			"timeouts": {
				"robot_state_timeout": self.state_timeout,
				"enemy_state_timeout": self.enemy_timeout,
			},
		}
