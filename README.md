# robot_vs

`robot_vs` 是一个 ROS1 多机器人红蓝对抗项目，当前包含两条并行技术线：

1. **ROS 对抗控制链路**：Manager 层 + Car 层 + Referee（裁判）
2. **MAS 多智能体链路**：LeaderAgent + CarAgent + 记忆系统（STM/LTM）+ 大模型服务

---

## 核心特性

- 支持 Gazebo + Rviz 仿真与真实部署迁移
- 红蓝双方独立决策与执行链路
- 命名空间 + TF 前缀隔离，支持多车并行
- `TaskCommand` / `RobotState` 闭环任务控制
- 支持 `GOTO / STOP / ATTACK / ROTATE` 动作
- 裁判节点统一处理可见敌人、命中判定、血量与弹药状态

---

## 快速开始

完整环境安装请先看 **[INSTALL.md](INSTALL.md)**。

```bash
# 1) 克隆到 catkin 工作空间
cd ~/catkin_ws/src
git clone https://github.com/Xqrion/robot_vs.git

# 2) 编译并加载环境
cd ~/catkin_ws
catkin_make
source devel/setup.bash

# 3) 启动管理层（红蓝 Manager + Referee，按 launch 配置）
roslaunch robot_vs managers.launch

# 4) 启动车辆层
roslaunch robot_vs cars.launch
```

---

## 文档索引

| 文档 | 内容 |
|---|---|
| [INSTALL.md](INSTALL.md) | 环境安装与部署 |
| [TECHNICAL.md](TECHNICAL.md) | ROS 技术文档（Manager/Car/Referee、消息流、可视化） |
| [TECHNICAL_MAS.md](TECHNICAL_MAS.md) | MAS 技术文档（LeaderAgent/CarAgent、记忆系统、LLM 服务、可视化） |

---

## 当前进展

| 模块 | 状态 |
|---|---|
| 多机器人仿真链路 | ✅ |
| Manager + Car + Skill | ✅ |
| Referee 裁判链路 | ✅ |
| MAS + LLM 双速率规划 | ✅（迭代中） |
| 真机稳定化与策略优化 | 🚧 |
