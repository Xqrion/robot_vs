#!/bin/bash

# 同时启动红蓝双方 LLM 规划服务。
# 依赖环境变量：
#   优先：LLM_API_KEY_RED / LLM_API_KEY_BLUE
#   回退：LLM_API_KEY
#   再回退：LLM_API

set -u

RED_PID=""
BLUE_PID=""

# 这个路径因人而异，注意看自己电脑
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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/../../scripts/AI/llm_manager.py"
CONFIG_PATH="${LLM_CONFIG_PATH:-$SCRIPT_DIR/llm_config.yaml}"
RED_PORT="${LLM_RED_PORT:-8001}"
BLUE_PORT="${LLM_BLUE_PORT:-8002}"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[start_llm_services] llm_manager.py not found: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -f "$CONFIG_PATH" ]; then
    echo "[start_llm_services] config not found: $CONFIG_PATH"
    exit 1
fi

if [ -z "${LLM_API_KEY_RED:-}" ] && [ -z "${LLM_API_KEY_BLUE:-}" ] && [ -z "${LLM_API_KEY:-}" ] && [ -z "${LLM_API:-}" ]; then
    echo "[start_llm_services] Missing API key env. Set one of:"
    echo "  LLM_API_KEY_RED / LLM_API_KEY_BLUE"
    echo "  LLM_API_KEY"
    echo "  LLM_API"
    exit 1
fi

port_listener_pid() {
    local port="$1"
    lsof -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null | head -n1
}

cleanup_stale_port() {
    local port="$1"
    local pid
    local cmd

    pid="$(port_listener_pid "$port")"
    if [ -z "$pid" ]; then
        return 0
    fi

    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if echo "$cmd" | grep -Fq "$PYTHON_SCRIPT"; then
        echo "[start_llm_services] Found stale llm_manager on port ${port} (pid=${pid}), terminating..."
        kill "$pid" 2>/dev/null || true
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        return 0
    fi

    echo "[start_llm_services] Port ${port} is occupied by non-llm process (pid=${pid}): ${cmd}"
    echo "[start_llm_services] Please free the port or set LLM_RED_PORT/LLM_BLUE_PORT."
    exit 1
}

cleanup_stale_port "$RED_PORT"
cleanup_stale_port "$BLUE_PORT"

echo "Starting LLM Planner Services for Red and Blue teams..."
echo "Config: $CONFIG_PATH"
echo "Ports: red=$RED_PORT blue=$BLUE_PORT"

python "$PYTHON_SCRIPT" --port "$RED_PORT" --side red --config "$CONFIG_PATH" &
RED_PID=$!
echo "Red Team LLM Service started with PID: $RED_PID"

python "$PYTHON_SCRIPT" --port "$BLUE_PORT" --side blue --config "$CONFIG_PATH" &
BLUE_PID=$!
echo "Blue Team LLM Service started with PID: $BLUE_PID"

cleanup() {
    local stopped_any=0
    local pid

    for pid in "$RED_PID" "$BLUE_PID"; do
        if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
            if [ "$stopped_any" -eq 0 ]; then
                echo "Terminating LLM Planner Services..."
                stopped_any=1
            fi
            kill "$pid" 2>/dev/null || true
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
    done
}

trap cleanup SIGINT SIGTERM EXIT

echo "Both services are running. Press Ctrl+C to stop."

wait "$RED_PID" 2>/dev/null
wait "$BLUE_PID" 2>/dev/null
