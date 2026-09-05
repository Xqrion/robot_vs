#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

"""Manager 层组件抽象基类。

定义 Observer / Formatter / Planner / Dispatcher 四大接口，
使 TeamManager 可以通过依赖注入使用任意实现了这些接口的组件。
"""

from abc import ABCMeta, abstractmethod


class BaseObserver(object):
    """战场状态观测器接口。"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_battle_state(self):
        """获取格式化的战场全局状态（含己方状态与敌方信息）。"""
        pass


class BaseFormatter(object):
    """战场状态格式化器接口。"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self, battle_state, team_color, my_cars):
        """将原始状态转换为规划器可读的格式。"""
        pass


class BasePlanner(object):
    """任务规划器接口。

    根据战场状态为每辆己方小车生成任务。
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def plan_tasks(self, battle_state):
        """根据战场状态生成任务字典 {robot_id: task_dict}。"""
        pass


class BaseDispatcher(object):
    """任务分发器接口。"""
    __metaclass__ = ABCMeta

    @abstractmethod
    def dispatch(self, tasks):
        """将任务字典发布到各小车对应的 ROS 话题。"""
        pass
