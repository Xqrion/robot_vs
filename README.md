# robot_vs

本仓库实现了多机器人红蓝对抗系统，采用 **Manager + Car Agent + Skill** 三层架构：
- **Manager 层**（`scripts/manager/`）：感知全局战场状态，调用 LLM 规划战术，通过 `TaskCommand` 消息向各小车下发任务。
- **Car Agent 层**（`scripts/car/`）：每辆小车运行一个独立的 `car_node.py`，接收任务并通过技能（Skill）执行动作，同时将 `RobotState` 反馈给 Manager。
- **Skill 系统**（`scripts/car/skills/`）：GoToSkill（导航）、StopSkill（刹车）、AttackSkill（攻击），实现任务的原子化执行。

详细架构说明与数据流图请参阅 → **[技术原理文档](TECHNICAL.md)**

---

## 演示 / Demo

> 🎬 演示图/视频即将更新，敬请期待……

---

## 功能特性

- 支持 **仿真环境** 与 **现实环境** 下的多机器人独立运行
- 采用 **红方 / 蓝方 两个阵营** 的对抗结构
  - 每个阵营有一个 Manager 节点负责 LLM 决策与任务分配
  - 每辆小车运行一个 Car Agent，通过技能系统执行 GOTO / STOP / ATTACK 三类动作
  - 小车携带 `mode` 字段区分待机 / 巡逻 / 攻击模式
- 基于 **命名空间 + TF 前缀** 实现多机话题隔离，防止冲突
- `TaskCommand` / `RobotState` 消息形成完整的任务下发与状态反馈闭环
- 仿真与现实话题结构保持一致，便于算法迁移
- 提供编辑好的 Rviz 可视化界面

---

## 快速开始

详细的环境搭建与部署步骤请参考 → **[环境配置文档](INSTALL.md)**

```bash
# 1. 克隆项目到 ROS 工作空间
cd ~/catkin_ws/src
git clone https://github.com/Xqrion/robot_vs.git

# 2. 编译
cd ~/catkin_ws && catkin_make && source devel/setup.bash

# 3. 启动 Manager（默认只启动红方；如需蓝方，取消注释 launch/manager/managers.launch 中对应节点）
roslaunch robot_vs managers.launch

# 4. 启动 Car Agent（红蓝各一辆；按需取消注释 launch/car/cars.launch 中多车配置）
roslaunch robot_vs cars.launch
```

> 两个 launch 文件均会自动加载对应的 YAML 配置文件，无需手动传参。  
> 如需修改巡逻点、队伍颜色等参数，请直接编辑 `config/` 目录下对应的 YAML 文件。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [环境配置](INSTALL.md) | 虚拟机搭建、ROS 安装、项目部署全流程 |
| [技术原理](TECHNICAL.md) | 系统架构、Manager/Car/Skill 详解、ROS 消息流、数据流图 |

---

## 项目状态

| 模块 | 状态 |
|------|------|
| 仿真环境（Gazebo + Rviz） | ✅ 已完成 |
| 红蓝阵营 Manager 框架 | ✅ 已完成 |
| Car Agent + Skill 系统 | ✅ 已完成 |
| 裁判系统对接 | 🚧 进行中 |
| 现实环境部署 | 🚧 进行中 |
