# 1. 任务元数据
uint32 task_id              # 任务流水号，递增
string action_type          # 动作类型：GOTO, STOP, ATTACK, ROTATE
string reason               # 大模型给出的战术意图（仅打印日志用）

# 2. 动作参数
float32 target_x            # 目标 X 坐标 (GOTO的终点，或 ATTACK 的瞄准点)
float32 target_y            # 目标 Y 坐标
float32 target_yaw
uint8 mode                  # 期望的模式 (0待机, 1巡逻, 2攻击)
float32 timeout             # 任务超时时间 (秒)，超时没完成就自动取消