#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import threading

import rospy
from robot_vs.msg import BattleMacroState
from robot_vs.msg import EnemyInfo
from robot_vs.msg import FireEvent
from robot_vs.msg import RobotState
from robot_vs.msg import TeamMacroState
from robot_vs.msg import VisibleEnemies
from nav_msgs.msg import OccupancyGrid


class RefereeNode(object):
    """全局唯一裁判节点。

    功能：
    1) 动态发现并订阅 /<ns>/robot_state 与 /<ns>/fire_event
    2) 维护全局状态（位姿、阵营、HP、生死）
    3) 处理开火命中判定并扣血
    4) 周期发布双方可见敌人列表
    """

    def __init__(self):
        self.loop_hz = float(rospy.get_param("~loop_hz", 10.0))
        self.discover_hz = float(rospy.get_param("~discover_hz", 1.0))

        self.default_hp = int(rospy.get_param("~default_hp", 100))
        self.default_ammo = float(rospy.get_param("~default_ammo", 50.0))
        self.fire_range = float(rospy.get_param("~fire_range", 5.0))
        self.hit_width = float(rospy.get_param("~hit_width", 0.5))
        self.fire_damage = int(rospy.get_param("~fire_damage", 20))
        self.vision_range = float(rospy.get_param("~vision_range", 4.0))

        self.fov_deg = float(rospy.get_param("~fov_deg", 120.0))
        self.fov_rad = math.radians(self.fov_deg)
        self.map_topic = str(rospy.get_param("~map_topic", "/map"))
        self.occ_threshold = int(rospy.get_param("~occ_threshold", 50))  # 0~100, >=阈值视为障碍
        self.block_unknown = bool(rospy.get_param("~block_unknown", True))  # -1 unknown 是否当障碍

        self._map_info = None
        self._map_data = None

        self._map_sub = rospy.Subscriber(self.map_topic, OccupancyGrid, self._on_map, queue_size=1)

        self._lock = threading.RLock()

        # dict[ns] = {"team", "x", "y", "yaw", "hp", "alive", "ammo"}
        self.global_states = {}

        self._robot_state_subs = {}
        self._fire_event_subs = {}

        self.red_enemy_pub = rospy.Publisher(
            "/red_manager/enemy_state", VisibleEnemies, queue_size=10
        )
        self.blue_enemy_pub = rospy.Publisher(
            "/blue_manager/enemy_state", VisibleEnemies, queue_size=10
        )
        self.macro_state_pub = rospy.Publisher(
            "/referee/macro_state", BattleMacroState, queue_size=10
        )

        rospy.loginfo(
            "RefereeNode initialized: loop_hz=%.1f discover_hz=%.1f fire_range=%.2f hit_width=%.2f fire_damage=%d vision_range=%.2f",
            self.loop_hz,
            self.discover_hz,
            self.fire_range,
            self.hit_width,
            self.fire_damage,
            self.vision_range,
        )

    @staticmethod
    def _normalize_ns(ns):
        return str(ns).strip().strip("/")

    @staticmethod
    def _parse_ns_from_topic(topic, suffix):
        if not topic or not topic.startswith("/"):
            return None
        if not topic.endswith(suffix):
            return None
        ns = topic[1 : -len(suffix)]
        ns = ns.strip("/")
        return ns if ns else None

    @staticmethod
    def _detect_team(ns):
        value = str(ns).lower()
        if "red" in value:
            return "red"
        if "blue" in value:
            return "blue"
        return "unknown"

    @staticmethod
    def _decode_team_code(team_code):
        """把 RobotState.team 的数值编码转为字符串阵营。"""
        try:
            code = int(team_code)
        except (TypeError, ValueError):
            return "unknown"

        # 约定来自 car 配置：0=red, 1=blue
        if code == 0:
            return "red"
        if code == 1:
            return "blue"
        return "unknown"

    @staticmethod
    def _quaternion_to_yaw(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _ensure_robot_record(self, ns):
        ns = self._normalize_ns(ns)
        if not ns:
            return None

        record = self.global_states.get(ns)
        if record is not None:
            return record

        record = {
            "team": self._detect_team(ns),
            "x": 0.0,
            "y": 0.0,
            "yaw": 0.0,
            "hp": int(self.default_hp),
            "ammo": float(self.default_ammo),
            "alive": True,
        }
        self.global_states[ns] = record
        rospy.loginfo("[referee] tracking robot: ns=%s team=%s", ns, record["team"])
        return record

    def _discover_and_subscribe(self):
        try:
            topics = rospy.get_published_topics()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "get_published_topics failed: %s", exc)
            return

        for topic, msg_type in topics:
            if topic.endswith("/robot_state") and msg_type == "robot_vs/RobotState":
                ns = self._parse_ns_from_topic(topic, "/robot_state")
                if not ns:
                    continue
                with self._lock:
                    self._ensure_robot_record(ns)
                    if ns not in self._robot_state_subs:
                        self._robot_state_subs[ns] = rospy.Subscriber(
                            topic,
                            RobotState,
                            self._on_robot_state,
                            callback_args=ns,
                            queue_size=20,
                        )
                        rospy.loginfo("[referee] subscribed robot_state: %s", topic)

            if topic.endswith("/fire_event") and msg_type == "robot_vs/FireEvent":
                ns = self._parse_ns_from_topic(topic, "/fire_event")
                if not ns:
                    continue
                with self._lock:
                    self._ensure_robot_record(ns)
                    if ns not in self._fire_event_subs:
                        self._fire_event_subs[ns] = rospy.Subscriber(
                            topic,
                            FireEvent,
                            self._on_fire_event,
                            callback_args=ns,
                            queue_size=50,
                        )
                        rospy.loginfo("[referee] subscribed fire_event: %s", topic)

    def _on_robot_state(self, msg, ns):
        with self._lock:
            record = self._ensure_robot_record(ns)
            if record is None:
                return

            team_from_msg = self._decode_team_code(msg.team)
            team_from_ns = self._detect_team(ns)
            if team_from_msg in ("red", "blue"):
                prev_team = record.get("team", "unknown")
                if team_from_ns in ("red", "blue") and team_from_ns != team_from_msg:
                    rospy.logwarn_throttle(
                        2.0,
                        "[referee] team mismatch: ns=%s ns_team=%s msg_team=%s",
                        ns,
                        team_from_ns,
                        team_from_msg,
                    )
                if prev_team != team_from_msg:
                    rospy.loginfo(
                        "[referee] team updated by RobotState: ns=%s %s->%s",
                        ns,
                        prev_team,
                        team_from_msg,
                    )
                record["team"] = team_from_msg
            elif record.get("team", "unknown") not in ("red", "blue"):
                # msg.team 无法解析时，才回退到命名空间推断。
                record["team"] = team_from_ns

            record["x"] = float(msg.pose.position.x)
            record["y"] = float(msg.pose.position.y)
            record["yaw"] = float(self._quaternion_to_yaw(msg.pose.orientation))

    def _ray_hit(self, shooter_x, shooter_y, shooter_yaw, target_x, target_y):
        dx = float(target_x) - float(shooter_x)
        dy = float(target_y) - float(shooter_y)
        dist = math.hypot(dx, dy)
        if dist <= 1e-6 or dist >= self.fire_range:
            return False

        dir_x = math.cos(shooter_yaw)
        dir_y = math.sin(shooter_yaw)

        forward = dx * dir_x + dy * dir_y
        if forward <= 0.0:
            return False

        # 2D 叉积模长=到射线垂距（方向向量已单位化）
        perp = abs(dx * dir_y - dy * dir_x)
        return perp < self.hit_width
    def _world_to_map(self, x, y):
        """世界坐标 -> 栅格坐标 (mx,my)，失败返回 None"""
        if self._map_info is None:
            return None
        origin = self._map_info.origin.position
        res = float(self._map_info.resolution)
        mx = int((x - origin.x) / res)
        my = int((y - origin.y) / res)
        if mx < 0 or my < 0 or mx >= self._map_info.width or my >= self._map_info.height:
            return None
        return mx, my

    def _grid_index(self, mx, my):
        return my * self._map_info.width + mx

    def _cell_blocked(self, mx, my):
        """该栅格是否视为障碍"""
        idx = self._grid_index(mx, my)
        val = int(self._map_data[idx])
        if val < 0:
            return bool(self.block_unknown)
        return val >= self.occ_threshold

    def _bresenham(self, x0, y0, x1, y1):
        """Bresenham 栅格线算法，yield (x,y)"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            yield x, y
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _has_line_of_sight(self, x0, y0, x1, y1):
        """用 /map 判断两点之间是否无遮挡。没有地图时默认 True。"""
        with self._lock:
            if self._map_info is None or self._map_data is None:
                return True
            p0 = self._world_to_map(x0, y0)
            p1 = self._world_to_map(x1, y1)
            if p0 is None or p1 is None:
                # 在地图外：保守做法可以返回 False；想放宽可返回 True
                return False
            x0m, y0m = p0
            x1m, y1m = p1

            first = True
            for mx, my in self._bresenham(x0m, y0m, x1m, y1m):
                if first:
                    first = False
                    continue  # 跳过起点格（避免自己所在格被膨胀层/噪声误判）
                if self._cell_blocked(mx, my):
                    return False
            return True

    def _on_fire_event(self, msg, topic_ns):
        shooter_ns = self._normalize_ns(msg.shooter_ns) or self._normalize_ns(topic_ns)
        if not shooter_ns:
            return

        with self._lock:
            shooter = self._ensure_robot_record(shooter_ns)
            if shooter is None:
                return

            shooter_team = shooter.get("team", "unknown")
            if shooter_team not in ("red", "blue"):
                rospy.logwarn_throttle(2.0, "[referee] unknown shooter team: %s", shooter_ns)
                return

            # 开火先进行弹药结算：无弹药则拦截，命中判定不再继续。
            if not shooter.get("alive", True):
                rospy.logwarn_throttle(2.0, "[referee] dead shooter fire blocked: %s", shooter_ns)
                return

            old_ammo = float(shooter.get("ammo", self.default_ammo))
            if old_ammo <= 0.0:
                rospy.logwarn_throttle(2.0, "[referee] fire blocked (no ammo): %s", shooter_ns)
                return
            shooter["ammo"] = max(0.0, old_ammo - 1.0)

            # 以 fire_event 的位姿作为射击真值。
            shooter["x"] = float(msg.x)
            shooter["y"] = float(msg.y)
            shooter["yaw"] = float(msg.yaw)

            enemy_team = "blue" if shooter_team == "red" else "red"
            for enemy_ns, enemy in self.global_states.items():
                if enemy_ns == shooter_ns:
                    continue
                if enemy.get("team") != enemy_team:
                    continue
                if not enemy.get("alive", True):
                    continue

                if self._ray_hit(
                    shooter["x"],
                    shooter["y"],
                    shooter["yaw"],
                    enemy.get("x", 0.0),
                    enemy.get("y", 0.0),
                ):
                    old_hp = int(enemy.get("hp", self.default_hp))
                    new_hp = max(0, old_hp - self.fire_damage)
                    enemy["hp"] = new_hp
                    enemy["alive"] = bool(new_hp > 0)

                    rospy.loginfo(
                        "[referee] hit: shooter=%s target=%s hp:%d->%d",
                        shooter_ns,
                        enemy_ns,
                        old_hp,
                        new_hp,
                    )

                    if old_hp > 0 and new_hp == 0:
                        rospy.loginfo("[referee] kill: shooter=%s target=%s", shooter_ns, enemy_ns)

    def _angle_diff(self, a, b):
        return math.atan2(math.sin(a - b), math.cos(a - b))
    
    def _build_visible_enemies(self, observer_team):
        enemy_team = "blue" if observer_team == "red" else "red"

        friendlies = []
        enemies = []
        for ns, state in self.global_states.items():
            if not state.get("alive", True):
                continue
            if state.get("team") == observer_team:
                friendlies.append((ns, state))
            elif state.get("team") == enemy_team:
                enemies.append((ns, state))

        visible = []
        half_fov = 0.5 * self.fov_rad
        for enemy_ns, enemy_state in enemies:
            ex = float(enemy_state.get("x", 0.0))
            ey = float(enemy_state.get("y", 0.0))

            seen = False
            for _, friendly_state in friendlies:
                fx = float(friendly_state.get("x", 0.0))
                fy = float(friendly_state.get("y", 0.0))
                fyaw = float(friendly_state.get("yaw", 0.0))

                dist = math.hypot(ex - fx, ey - fy)
                if dist > self.vision_range:
                 continue

                bearing = math.atan2(ey - fy, ex - fx)
                if abs(self._angle_diff(bearing, fyaw)) > half_fov:
                 continue

                if not self._has_line_of_sight(fx, fy, ex, ey):
                 continue
                seen = True
                break

            if seen:
                info = EnemyInfo()
                info.robot_ns = enemy_ns
                info.x = ex
                info.y = ey
                info.hp = int(enemy_state.get("hp", self.default_hp))
                visible.append(info)

        msg = VisibleEnemies()
        msg.enemies = visible
        return msg

    def _publish_visible_enemies(self):
        with self._lock:
            red_msg = self._build_visible_enemies("red")
            blue_msg = self._build_visible_enemies("blue")

        self.red_enemy_pub.publish(red_msg)
        self.blue_enemy_pub.publish(blue_msg)

    def _build_team_macro_state(self, team):
        msg = TeamMacroState()
        msg.team = str(team)

        total_hp = 0
        total_ammo = 0.0
        alive_count = 0
        dead_count = 0

        for ns in sorted(self.global_states.keys()):
            state = self.global_states.get(ns, {})
            if state.get("team") != team:
                continue

            hp = int(state.get("hp", self.default_hp))
            ammo = float(state.get("ammo", self.default_ammo))
            alive = bool(state.get("alive", True) and hp > 0)

            msg.robot_ns.append(ns)
            msg.hp.append(hp)
            msg.ammo.append(ammo)
            msg.alive.append(alive)

            total_hp += hp
            total_ammo += ammo
            if alive:
                alive_count += 1
            else:
                dead_count += 1

        msg.total_hp = int(total_hp)
        msg.total_ammo = float(total_ammo)
        msg.alive_count = int(alive_count)
        msg.dead_count = int(dead_count)
        return msg

    def _publish_macro_state(self):
        with self._lock:
            red = self._build_team_macro_state("red")
            blue = self._build_team_macro_state("blue")

        msg = BattleMacroState()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.red = red
        msg.blue = blue
        self.macro_state_pub.publish(msg)

    def run(self):
        main_rate = rospy.Rate(self.loop_hz)
        discover_interval = 1.0 / self.discover_hz if self.discover_hz > 0.0 else 1.0
        last_discover = 0.0

        while not rospy.is_shutdown():
            now = rospy.Time.now().to_sec()
            if now - last_discover >= discover_interval:
                self._discover_and_subscribe()
                last_discover = now

            self._publish_visible_enemies()
            self._publish_macro_state()
            main_rate.sleep()

    def _on_map(self, msg):
        with self._lock:
            self._map_info = msg.info
            self._map_data = msg.data  # tuple/list of int8

def main():
    rospy.init_node("referee_node", anonymous=False)

    node = RefereeNode()
    node.run()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
