std_msgs/Header header

# 1. 身份与基础信息
string robot_ns
uint8 team

# 2. 战斗与生存状态 (供裁判系统/大模型参考)
float32 hp
float32 ammo
bool alive
bool in_combat

# 3. 运动学状态 (我在哪)
geometry_msgs/Pose pose
geometry_msgs/Twist twist

# 4. 任务执行反馈 (★新增：沟通 Manager 的核心)
uint32 current_task_id      # 我正在执行的、或者刚完成的任务ID
string current_action       # 我当前在干嘛 (GOTO, STOP, ATTACK)
string task_status          # 执行结果: "RUNNING", "SUCCESS", "FAILED"
uint8 mode                  # 我当前的物理模式 (0待机, 1巡逻, 2攻击)