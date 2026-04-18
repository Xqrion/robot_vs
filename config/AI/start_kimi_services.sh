#!/bin/bash

##这个路径因人而异，注意看自己电脑
__conda_setup="$('/home/xqrion/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/xqrion/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/xqrion/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/xqrion/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate robotvs

## 这个脚本用于同时启动红蓝双方的kimi服务，并且在接收到 SIGINT（如 Ctrl+C）时能够优雅地关闭这两个服务。
## 注意：确保在运行此脚本之前，已经正确配置了必要的环境变量（如 KIMI_API_KEY_RED 和 KIMI_API_KEY_BLUE）。
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/../../scripts/AI/kimi_manager.py"

echo "Starting Kimi Planner Services for Red and Blue teams..."

# 后台启动红方服务
python "$PYTHON_SCRIPT" --port 8001 --side red &
RED_PID=$!
echo "Red Team Kimi Service started with PID: $RED_PID"

# 后台启动蓝方服务
python "$PYTHON_SCRIPT" --port 8002 --side blue &
BLUE_PID=$!
echo "Blue Team Kimi Service started with PID: $BLUE_PID"

# trap SIGINT 时 kill 两个进程，不再使用 exit 防止在 source 运行时关闭当前终端
trap "echo 'Terminating Kimi Planner Services...'; kill $RED_PID $BLUE_PID 2>/dev/null" SIGINT SIGTERM

echo "Both services are running. Press Ctrl+C to stop."

# wait 等待，屏蔽因为中断抛出的错误
wait $RED_PID 2>/dev/null
wait $BLUE_PID 2>/dev/null