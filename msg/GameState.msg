# 比赛当前状态（周期性发布，约 1 Hz）
string status        # IDLE / PLAYING / FINISHED
float32 elapsed      # 已过去秒数
float32 time_limit   # 时间限制（0 = 不限时）
string winner        # "" / "red" / "blue" / "draw"
string reason        # "" / "all_enemy_dead" / "timeout" / "manual_stop"
