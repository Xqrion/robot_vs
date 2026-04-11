#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import threading

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from robot_vs.msg import RobotState, BattleMacroState, FireEvent, VisibleEnemies


class VisualizerNode:
    """可视化节点：显示血条、弹药文字、彩色底盘以及弹道命中区域"""

    def __init__(self):
        self._lock = threading.RLock()
        self.robot_info = {}          # 存储机器人信息：位姿、血量、弹药、存活、队伍
        self.visible_info = {}        # 可见性字典：ns -> bool（是否被敌方发现）

        # 弹道显示参数
        self._fire_subs = {}
        self.fire_range = rospy.get_param("~fire_range", 5.0)
        self.hit_width = rospy.get_param("~hit_width", 0.5)          # 命中判定半宽
        self.trajectory_lifetime = rospy.get_param("~trajectory_lifetime", 0.5)
        self.show_trajectory_line = rospy.get_param("~show_trajectory_line", False)  # 是否同时显示中心线

        # 底盘显示参数
        self.chassis_enabled = rospy.get_param("~chassis_enabled", True)
        self.chassis_radius = rospy.get_param("~chassis_radius", 0.35)
        self.chassis_height = rospy.get_param("~chassis_height", 0.08)
        self.chassis_z = rospy.get_param("~chassis_z", 0.05)

        # 订阅机器人状态和开火事件
        self.state_subs = {}
        discover_hz = rospy.get_param("~discover_hz", 2.0)   # 默认 2 Hz（周期 0.5 秒）
        self.discover_timer = rospy.Timer(rospy.Duration(1.0/discover_hz), self.discover_topics)

        # 订阅全局状态和可见性信息
        self.macro_sub = rospy.Subscriber("/referee/macro_state", BattleMacroState, self.macro_callback)
        self.red_enemy_sub = rospy.Subscriber("/red_manager/enemy_state", VisibleEnemies, self.red_enemy_callback)
        self.blue_enemy_sub = rospy.Subscriber("/blue_manager/enemy_state", VisibleEnemies, self.blue_enemy_callback)

        # 发布器
        self.marker_pub = rospy.Publisher("/health_markers", MarkerArray, queue_size=10)
        self.traj_pub = rospy.Publisher("/trajectory_markers", MarkerArray, queue_size=10)
        self.chassis_pub = rospy.Publisher("/chassis_markers", MarkerArray, queue_size=10)

        # 显示刷新定时器
        display_hz = rospy.get_param("~display_hz", 20.0)
        self.display_timer = rospy.Timer(rospy.Duration(1.0 / display_hz), self.publish_markers)

        rospy.loginfo("VisualizerNode started: fire_range=%.2f hit_width=%.2f display_hz=%.1f",
                      self.fire_range, self.hit_width, display_hz)

    def discover_topics(self, event=None):
        """动态发现 /robot_state 和 /fire_event 主题并订阅"""
        try:
            topics = rospy.get_published_topics()
        except Exception as e:
            rospy.logwarn_throttle(5.0, "Failed to get topics: %s", e)
            return

        with self._lock:
            for topic, msg_type in topics:
                # 机器人状态
                if topic.endswith("/robot_state") and msg_type == "robot_vs/RobotState":
                    ns = self._parse_ns(topic, "/robot_state")
                    if ns and ns not in self.state_subs:
                        self.state_subs[ns] = rospy.Subscriber(
                            topic, RobotState, self.robot_state_cb, callback_args=ns, queue_size=10
                        )
                        rospy.loginfo("Subscribed to %s (ns=%s)", topic, ns)

                # 开火事件（弹道）
                if topic.endswith("/fire_event") and msg_type == "robot_vs/FireEvent":
                    ns = self._parse_ns(topic, "/fire_event")
                    if ns and ns not in self._fire_subs:
                        self._fire_subs[ns] = rospy.Subscriber(
                            topic, FireEvent, self.fire_event_cb, callback_args=ns, queue_size=20
                        )
                        rospy.loginfo("Subscribed to fire_event: %s", topic)

    @staticmethod
    def _parse_ns(topic, suffix):
        """从话题名中提取命名空间"""
        if not topic.startswith("/") or not topic.endswith(suffix):
            return None
        ns = topic[1:-len(suffix)].strip("/")
        return ns if ns else None

    def robot_state_cb(self, msg, ns):
        """接收机器人位姿信息"""
        with self._lock:
            if ns not in self.robot_info:
                self.robot_info[ns] = {}
            self.robot_info[ns]["x"] = msg.pose.position.x
            self.robot_info[ns]["y"] = msg.pose.position.y
            q = msg.pose.orientation
            siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            self.robot_info[ns]["yaw"] = math.atan2(siny_cosp, cosy_cosp)
            self.robot_info[ns]["last_seen"] = rospy.Time.now()

    def macro_callback(self, msg):
        """接收宏观战况，更新血量、弹药、存活状态及队伍归属"""
        with self._lock:
            # 红队
            for ns, hp, ammo in zip(msg.red.robot_ns, msg.red.hp, msg.red.ammo):
                if ns not in self.robot_info:
                    self.robot_info[ns] = {}
                self.robot_info[ns]["hp"] = hp
                self.robot_info[ns]["ammo"] = ammo
                self.robot_info[ns]["alive"] = msg.red.alive[msg.red.robot_ns.index(ns)] if ns in msg.red.robot_ns else (hp > 0)
                self.robot_info[ns]["team"] = "red"
            # 蓝队
            for ns, hp, ammo in zip(msg.blue.robot_ns, msg.blue.hp, msg.blue.ammo):
                if ns not in self.robot_info:
                    self.robot_info[ns] = {}
                self.robot_info[ns]["hp"] = hp
                self.robot_info[ns]["ammo"] = ammo
                self.robot_info[ns]["alive"] = msg.blue.alive[msg.blue.robot_ns.index(ns)] if ns in msg.blue.robot_ns else (hp > 0)
                self.robot_info[ns]["team"] = "blue"

    def red_enemy_callback(self, msg):
        """红方看到的敌方（蓝队）列表 -> 标记这些蓝车被红方发现"""
        with self._lock:
            # 先将所有蓝车的可见性设为 False
            for ns, info in self.robot_info.items():
                if info.get("team") == "blue":
                    self.visible_info[ns] = False
            # 将被发现的蓝车设为 True
            for enemy in msg.enemies:
                ns = enemy.robot_ns
                if ns in self.robot_info and self.robot_info[ns].get("team") == "blue":
                    self.visible_info[ns] = True

    def blue_enemy_callback(self, msg):
        """蓝方看到的敌方（红队）列表 -> 标记这些红车被蓝方发现"""
        with self._lock:
            # 先将所有红车的可见性设为 False
            for ns, info in self.robot_info.items():
                if info.get("team") == "red":
                    self.visible_info[ns] = False
            # 将被发现的红车设为 True
            for enemy in msg.enemies:
                ns = enemy.robot_ns
                if ns in self.robot_info and self.robot_info[ns].get("team") == "red":
                    self.visible_info[ns] = True

    def fire_event_cb(self, msg, ns):
        """收到开火事件，生成命中区域矩形 Marker（半透明立方体）"""
        start_x = msg.x
        start_y = msg.y
        yaw = msg.yaw
        length = self.fire_range
        half_width = self.hit_width          # 矩形半宽，实际总宽 = 2 * hit_width
        thickness = 0.02                     # 矩形厚度（视觉高度）

        # 计算矩形中心点（位于射击点沿方向 length/2 处）
        center_x = start_x + (length / 2.0) * math.cos(yaw)
        center_y = start_y + (length / 2.0) * math.sin(yaw)
        center_z = 0.05                      # 贴近地面

        # 创建矩形立方体 Marker
        rect_marker = Marker()
        rect_marker.header.frame_id = "map"
        rect_marker.header.stamp = rospy.Time.now()
        rect_marker.ns = "trajectory_rect"
        rect_marker.id = int((rospy.Time.now().to_sec() * 1000) % 1000000)
        rect_marker.type = Marker.CUBE
        rect_marker.action = Marker.ADD
        rect_marker.pose.position.x = center_x
        rect_marker.pose.position.y = center_y
        rect_marker.pose.position.z = center_z
        rect_marker.scale.x = length
        rect_marker.scale.y = 2.0 * half_width   # 总宽
        rect_marker.scale.z = thickness
        # 设置颜色：橙色半透明
        rect_marker.color.a = 0.4
        rect_marker.color.r = 1.0
        rect_marker.color.g = 0.5
        rect_marker.color.b = 0.0
        rect_marker.lifetime = rospy.Duration(self.trajectory_lifetime)

        # 设置方向（绕 Z 轴旋转 yaw）
        from tf.transformations import quaternion_from_euler
        q = quaternion_from_euler(0, 0, yaw)
        rect_marker.pose.orientation.x = q[0]
        rect_marker.pose.orientation.y = q[1]
        rect_marker.pose.orientation.z = q[2]
        rect_marker.pose.orientation.w = q[3]

        traj_array = MarkerArray()
        traj_array.markers.append(rect_marker)

        # 可选：同时显示中心线（便于观察朝向）
        if self.show_trajectory_line:
            end_x = start_x + length * math.cos(yaw)
            end_y = start_y + length * math.sin(yaw)
            line_marker = Marker()
            line_marker.header.frame_id = "map"
            line_marker.header.stamp = rospy.Time.now()
            line_marker.ns = "trajectory_line"
            line_marker.id = rect_marker.id + 1000
            line_marker.type = Marker.LINE_STRIP
            line_marker.action = Marker.ADD
            line_marker.scale.x = 0.01
            line_marker.color.a = 0.8
            line_marker.color.r = 1.0
            line_marker.color.g = 1.0
            line_marker.color.b = 1.0
            line_marker.lifetime = rospy.Duration(self.trajectory_lifetime)
            start_point = Point(start_x, start_y, 0.12)
            end_point = Point(end_x, end_y, 0.12)
            line_marker.points = [start_point, end_point]
            traj_array.markers.append(line_marker)

        self.traj_pub.publish(traj_array)

    def publish_markers(self, event=None):
        """发布血条、弹药文字以及彩色底盘"""
        now = rospy.Time.now()
        health_markers = MarkerArray()
        chassis_markers = MarkerArray()

        # 清除旧的健康显示 Marker (ns="health")
        clear_health = Marker()
        clear_health.header.frame_id = "map"
        clear_health.header.stamp = now
        clear_health.ns = "health"
        clear_health.id = 0
        clear_health.action = Marker.DELETEALL
        health_markers.markers.append(clear_health)

        # 清除旧的底盘 Marker (ns="chassis")
        if self.chassis_enabled:
            clear_chassis = Marker()
            clear_chassis.header.frame_id = "map"
            clear_chassis.header.stamp = now
            clear_chassis.ns = "chassis"
            clear_chassis.id = 0
            clear_chassis.action = Marker.DELETEALL
            chassis_markers.markers.append(clear_chassis)

        marker_id = 0
        chassis_id = 0

        with self._lock:
            for ns, info in self.robot_info.items():
                if not info.get("alive", True):
                    continue
                last_seen = info.get("last_seen", rospy.Time(0))
                if (now - last_seen).to_sec() > 10.0:
                    continue
                x = info.get("x", 0.0)
                y = info.get("y", 0.0)
                hp = info.get("hp", 0)
                ammo = info.get("ammo", 0.0)
                team = info.get("team", "unknown")
                if hp <= 0:
                    continue

                # 血条（绿色到红色渐变）
                bar = Marker()
                bar.header.frame_id = "map"
                bar.header.stamp = now
                bar.ns = "health"
                bar.id = marker_id
                marker_id += 1
                bar.type = Marker.CUBE
                bar.action = Marker.ADD
                bar.pose.position.x = x
                bar.pose.position.y = y + 0.15
                bar.pose.position.z = 0.3
                max_len = 0.5
                bar.scale.x = max_len * (hp / 100.0)
                bar.scale.y = 0.08
                bar.scale.z = 0.05
                bar.color.a = 0.8
                if hp > 70:
                    bar.color.g = 1.0
                    bar.color.r = 0.0
                elif hp > 30:
                    bar.color.r = 1.0
                    bar.color.g = 1.0
                    bar.color.b = 0.0
                else:
                    bar.color.r = 1.0
                health_markers.markers.append(bar)

                # 血量文字
                text_hp = Marker()
                text_hp.header.frame_id = "map"
                text_hp.header.stamp = now
                text_hp.ns = "health"
                text_hp.id = marker_id
                marker_id += 1
                text_hp.type = Marker.TEXT_VIEW_FACING
                text_hp.action = Marker.ADD
                text_hp.pose.position.x = x
                text_hp.pose.position.y = y + 0.15
                text_hp.pose.position.z = 0.45
                text_hp.scale.z = 0.12
                text_hp.color.a = 1.0
                text_hp.color.r = 0.0
                text_hp.color.g = 0.0
                text_hp.color.b = 0.0
                text_hp.text = "HP: {}".format(hp)
                health_markers.markers.append(text_hp)

                # 弹药文字
                text_ammo = Marker()
                text_ammo.header.frame_id = "map"
                text_ammo.header.stamp = now
                text_ammo.ns = "health"
                text_ammo.id = marker_id
                marker_id += 1
                text_ammo.type = Marker.TEXT_VIEW_FACING
                text_ammo.action = Marker.ADD
                text_ammo.pose.position.x = x
                text_ammo.pose.position.y = y - 0.15
                text_ammo.pose.position.z = 0.35
                text_ammo.scale.z = 0.09
                text_ammo.color.a = 1.0
                text_ammo.color.g = 0.0
                text_ammo.color.r = 0.0
                text_ammo.text = "Ammo: {:.0f}".format(ammo)
                health_markers.markers.append(text_ammo)

                # 彩色底盘（独立 Marker）
                if not self.chassis_enabled:
                    continue

                # 确定颜色和透明度
                if team == "red":
                    base_color = (1.0, 0.0, 0.0)  # 红
                elif team == "blue":
                    base_color = (0.0, 0.0, 1.0)  # 蓝
                else:
                    continue

                # 可见性：默认半透明，被敌方发现则加深
                is_visible = self.visible_info.get(ns, False)
                alpha = 0.6 if is_visible else 0.3

                chassis = Marker()
                chassis.header.frame_id = "map"
                chassis.header.stamp = now
                chassis.ns = "chassis"
                chassis.id = chassis_id
                chassis_id += 1
                chassis.type = Marker.CYLINDER
                chassis.action = Marker.ADD
                chassis.pose.position.x = x
                chassis.pose.position.y = y
                chassis.pose.position.z = self.chassis_z
                chassis.scale.x = self.chassis_radius * 2.0   # 直径
                chassis.scale.y = self.chassis_radius * 2.0
                chassis.scale.z = self.chassis_height
                chassis.color.r = base_color[0]
                chassis.color.g = base_color[1]
                chassis.color.b = base_color[2]
                chassis.color.a = alpha
                chassis.lifetime = rospy.Duration(0.2)  # 及时刷新
                chassis_markers.markers.append(chassis)

        # 发布
        self.marker_pub.publish(health_markers)
        if self.chassis_enabled:
            self.chassis_pub.publish(chassis_markers)


if __name__ == "__main__":
    rospy.init_node("viz_node")
    node = VisualizerNode()
    rospy.spin()