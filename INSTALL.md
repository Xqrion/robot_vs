# Ubuntu 18.04 虚拟机环境搭建与 robot_vs 项目部署说明书

## 目录

- [1. 项目说明](#1-项目说明)
- [2. 环境准备](#2-环境准备)
- [3. 使用虚拟机安装 Ubuntu 18.04](#3-使用虚拟机安装-ubuntu-1804)
  - [3.1 下载 Ubuntu 18.04 镜像](#31-下载-ubuntu-1804-镜像)
  - [3.2 创建 Ubuntu 虚拟机](#32-创建-ubuntu-虚拟机)
  - [3.3 安装 Ubuntu 18.04 系统](#33-安装-ubuntu-1804-系统)
  - [3.4 安装完成后的注意事项（必读）](#34-安装完成后的注意事项必读)
- [4. 使用鱼香 ROS 一键安装 ROS Melodic](#4-使用鱼香-ros-一键安装-ros-melodic)
- [5. 安装 VS Code](#5-安装-vs-code)
- [6. 安装 Git](#6-安装-git)
- [7. 创建 ROS 工作空间](#7-创建-ros-工作空间)
- [8. 安装 Conda 并配置 Python 环境](#8-安装-conda-并配置-python-环境)
- [9. 下载并本地部署 robot_vs 项目](#9-下载并本地部署-robot_vs-项目)
- [10. 安装 TurtleBot3](#10-安装-turtlebot3)
- [11. 运行程序](#11-运行程序)
- [12. 常见问题及解决方法](#12-常见问题及解决方法)
- [13. 核心命令一键汇总](#13-核心命令一键汇总)

---

## 1. 项目说明

本文档介绍如何在 **虚拟机** 中安装 **Ubuntu 18.04**，并使用 **鱼香 ROS 一键安装工具** 完成 ROS 环境搭建，随后安装 **VS Code**、**Git**，将 GitHub 项目 `robot_vs` 下载到本地并完成部署，安装 **TurtleBot3** 相关功能包，最后运行程序。

> **特别注意**：Ubuntu 18.04 安装完成后，**千万不要先执行系统更新或升级操作（apt update/upgrade）**，以避免出现 ROS Melodic 与系统软件最新版本之间的兼容性问题。

---

## 2. 环境准备

- **虚拟机软件**：VMware Workstation 或 VirtualBox
- **系统镜像**：Ubuntu 18.04 LTS
- **硬件推荐**：分配给虚拟机至少 4GB 内存（推荐 8GB），至少 2 核 CPU，至少 40GB 磁盘空间。
- **网络**：需要保证虚拟机能够正常连接互联网。

---

## 3. 使用虚拟机安装 Ubuntu 18.04

### 3.1 下载 Ubuntu 18.04 镜像

打开 Ubuntu 官方历史版本下载页面：
[https://releases.ubuntu.com/18.04/](https://releases.ubuntu.com/18.04/)

下载文件：`ubuntu-18.04.6-desktop-amd64.iso`

### 3.2 创建 Ubuntu 虚拟机

以 **VMware Workstation** 为例：
1. 点击 **创建新的虚拟机**，选择 **典型（推荐）**。
2. 选择 **安装程序光盘映像文件（ISO）**，选中刚才下载的 Ubuntu 18.04 镜像。
3. 客户机操作系统选择 `Linux`，版本选择 `Ubuntu 64-bit`。
4. 设置虚拟机名称和保存路径。
5. 磁盘大小建议设置为 **40GB 以上**。
6. 点击 **自定义硬件**：
   - 内存：建议设为 **8GB**（最低 4GB）
   - 处理器：建议 **2核或4核**
   - 显示器：勾选 **加速 3D 图形**（对后续运行 Gazebo 仿真非常重要）
7. 完成创建。

### 3.3 安装 Ubuntu 18.04 系统

启动虚拟机，按照界面提示进行安装：
1. 欢迎界面选择 `Install Ubuntu`。
2. 语言和键盘布局保持默认或选择习惯语言。
3. 安装类型选择 **Normal installation**。
4. 磁盘选项选择 **Erase disk and install Ubuntu**（仅清空虚拟机磁盘，不影响本机）。
5. 设置用户名、计算机名和密码，等待安装完成并重启系统。

### 3.4 安装完成后的注意事项（必读）

系统重启进入桌面后，系统可能会弹窗提示“是否升级系统”，**请一律选择“不升级/关闭”**。

同时，**绝对不要**在终端手动执行以下命令：
```bash
sudo apt update
sudo apt upgrade -y
```
保持原始环境能够最大程度保证后续 ROS Melodic 及仿真环境的稳定安装。

---

## 4. 使用鱼香 ROS 一键安装 ROS Melodic

Ubuntu 18.04 对应的 ROS 版本为 **ROS Melodic**。为解决国内网络问题，这里使用鱼香 ROS 一键安装。

打开终端（`Ctrl + Alt + T`），执行以下命令：

```bash
wget http://fishros.com/install -O fishros && . fishros
```

进入安装菜单后，按照提示依次选择：
1. 选择 **[1] 一键安装 ROS 及配置环境**
2. 选择版本 **ROS Melodic (Ubuntu 18.04)**
3. 选择 **桌面完整版 (Desktop-Full)**
4. 同意自动配置软件源和环境变量

安装完成后，关闭当前终端，**重新打开一个新终端**，执行验证：

```bash
roscore
```
如果出现 `started core service [/rosout]` 等日志且未报错，说明 ROS 安装成功。按 `Ctrl + C` 退出。

---

## 5. 安装 VS Code（也可以用鱼香ros一键安装)

VS Code 用于后续查看和编写代码。

在终端中执行以下命令（如果提示未找到 snap，会自动通过第二条命令安装）：

```bash
sudo apt install -y snapd
sudo snap install --classic code
```

验证安装（在终端中输入以下命令，若能弹出 VS Code 界面即为成功）：

```bash
code
```

---

## 6. 安装 Git

Git 用于从 GitHub 拉取代码。

```bash
sudo apt install -y git
```

验证安装：

```bash
git --version
```

配置 Git 用户信息（建议）：

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

---

## 7. 创建 ROS 工作空间

ROS 项目需要放置在特定的工作空间（Workspace）中编译。

```bash
# 创建 src 目录
mkdir -p ~/catkin_ws/src

# 进入工作空间并初始化编译
cd ~/catkin_ws
catkin_make

# 将工作空间环境变量写入 .bashrc
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## 8. 安装 Conda 并配置 Python 环境

### 8.1 安装 Miniconda

```bash
# 下载 Miniconda 安装脚本（Python 3.9 版本）
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh

# 运行安装（按提示同意协议，默认安装路径即可）
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh

# 重新加载 bash 配置或新开终端
source ~/.bashrc
```

### 8.2 创建虚拟环境并安装依赖

```bash
# 进入项目目录（假设已 clone 到 src 下）
cd ~/robo2026_ws/src/robot_vs

# 用 conda 创建名为 robotvs 的 Python 3.9 环境
conda create -n robotvs python=3.9 -y

# 激活环境
conda activate robotvs

# 安装 requirements.txt 中的依赖
pip install -r requirements.txt
```

---

## 9. 下载并本地部署 robot_vs 项目

### 9.1 克隆项目

进入工作空间的 `src` 目录，拉取 GitHub 上的代码：

```bash
cd ~/catkin_ws/src
git clone https://github.com/Xqrion/robot_vs.git
```

### 9.2 安装项目依赖

返回工作空间根目录，使用 `rosdep` 自动安装该项目所需的依赖包：

```bash
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 9.3 编译项目

执行编译并刷新环境变量：

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

---

## 10. 安装 TurtleBot3

该项目通常需要依赖 TurtleBot3 的仿真环境，需安装相关功能包：

```bash
# 安装 TurtleBot3 基础包和仿真包
sudo apt install -y ros-melodic-turtlebot3 ros-melodic-turtlebot3-simulations

# 设置默认机器人型号（常用型号为 burger）
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
```

---

## 11. 运行程序

运行该项目通常需要打开 **三个不同的终端窗口** 进行配合。

### 终端 1：启动 ROS 节点管理器

```bash
roscore
```
*(保持此终端不要关闭)*

### 终端 2：测试 TurtleBot3 仿真环境

打开新终端，启动 Gazebo 仿真世界：

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch turtlebot3_gazebo turtlebot3_world.launch
```
*(等待 Gazebo 界面完全加载出来)*

### 终端 3：启动 robot_vs 项目

打开新终端，启动项目核心代码。
*注：具体启动的 launch 文件名需根据 `robot_vs` 项目内部结构决定，此处假设为 `main.launch`*

```bash
source ~/catkin_ws/devel/setup.bash
roslaunch robot_vs main.launch
```

> **补充：如果项目是通过 Python 脚本运行的**
> 如果项目不是用 `.launch` 而是直接运行 `.py` 脚本，需要先赋予权限再运行：
> ```bash
> chmod +x ~/catkin_ws/src/robot_vs/scripts/*.py
> cd ~/catkin_ws && catkin_make && source devel/setup.bash
> rosrun robot_vs <对应的脚本名称.py>
> ```

---

## 12. 常见问题及解决方法

1. **`catkin_make` 编译失败**
   - 检查报错提示中缺失了什么包，使用 `sudo apt install ros-melodic-<包名>` 手动安装。
   - 确保你在 `~/catkin_ws` 目录下执行编译命令，而不是在 `src` 目录下。

2. **提示找不到 `robot_vs` 包**
   - 确保代码放在 `~/catkin_ws/src` 中。
   - 确保执行过 `catkin_make` 且没报错。
   - 确保执行了 `source ~/catkin_ws/devel/setup.bash`。

3. **虚拟机运行 Gazebo 仿真特别卡顿**
   - 在虚拟机关机状态下，进入虚拟机设置 -> 显示器 -> 勾选“**加速 3D 图形**”。
   - 提升虚拟机内存到 8GB。

---

## 13. 核心命令一键汇总

如果你熟悉流程，可以直接按顺序复制以下代码块执行：

```bash
# 1. 鱼香ROS安装
wget http://fishros.com/install -O fishros && . fishros

# 2. 安装 VS Code 和 Git
sudo apt install -y snapd git
sudo snap install --classic code

# 3. 创建工作空间并配置环境
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws && catkin_make
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

# 4. 下载并编译 robot_vs
cd ~/catkin_ws/src
git clone https://github.com/Xqrion/robot_vs.git
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash

# 5. 安装 TurtleBot3
sudo apt install -y ros-melodic-turtlebot3 ros-melodic-turtlebot3-simulations
echo "export TURTLEBOT3_MODEL=burger" >> ~/.bashrc
source ~/.bashrc
```
```
