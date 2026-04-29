#!/bin/bash

# Start MAS dual-port service (one process):
# - red side on RED_PORT (default 8001)
# - blue side on BLUE_PORT (default 8002)

set -u

MAS_PID=""

# Configure conda shell hooks.
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

CONDA_ENV="${MAS_CONDA_ENV:-robotvs}"
conda activate "$CONDA_ENV" >/dev/null 2>&1 || {
    echo "[start_mas_services] Failed to activate conda env: $CONDA_ENV"
    exit 1
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/../../scripts/MAS/llm_server.py"
CONFIGS_ROOT="${MAS_CONFIGS_ROOT:-$SCRIPT_DIR/../../scripts/MAS}"

HOST="${MAS_HOST:-0.0.0.0}"
RED_PORT="${MAS_RED_PORT:-${LLM_RED_PORT:-8001}}"
BLUE_PORT="${MAS_BLUE_PORT:-${LLM_BLUE_PORT:-8002}}"
LOG_LEVEL="${MAS_LOG_LEVEL:-info}"

# <<< switch to debug <<<
# Prompt trace settings:
# - MAS_LOG_PROMPTS: 1 to enable prompt trace logging (default on for debugging)
# - MAS_PROMPT_LOG_FILE: output file for formatted prompt/response trace
# - MAS_PROMPT_LOG_CONSOLE: 1 to also print full trace in terminal (default off)
# - MAS_SPLIT_PROMPT_LOGS: 1 to split into leader/car log files
# - MAS_PROMPT_LOG_PER_RUN: 1 to generate new files per run
# - MAS_RUN_ID: run label used in output filenames (auto-generated when empty)
export MAS_LOG_PROMPTS="${MAS_LOG_PROMPTS:-1}"
export MAS_PROMPT_LOG_FILE="${MAS_PROMPT_LOG_FILE:-$SCRIPT_DIR/../../debug/mas_llm_trace.log}"
export MAS_PROMPT_LOG_CONSOLE="${MAS_PROMPT_LOG_CONSOLE:-0}"
export MAS_SPLIT_PROMPT_LOGS="${MAS_SPLIT_PROMPT_LOGS:-1}"
export MAS_PROMPT_LOG_PER_RUN="${MAS_PROMPT_LOG_PER_RUN:-1}"
export MAS_RUN_ID="${MAS_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# Keep prompts_2 as current default, but allow override from environment.
export MAS_PROMPTS_FILE="${MAS_PROMPTS_FILE:-prompts_2.yaml}"

if [ "$MAS_LOG_PROMPTS" = "1" ]; then
    mkdir -p "$(dirname "$MAS_PROMPT_LOG_FILE")"
fi

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[start_mas_services] MAS server script not found: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -d "$CONFIGS_ROOT" ]; then
    echo "[start_mas_services] MAS configs root not found: $CONFIGS_ROOT"
    exit 1
fi

if [ "$RED_PORT" = "$BLUE_PORT" ]; then
    echo "[start_mas_services] RED_PORT and BLUE_PORT must be different."
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
        echo "[start_mas_services] Found stale MAS server on port ${port} (pid=${pid}), terminating..."
        kill "$pid" 2>/dev/null || true
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
        return 0
    fi

    echo "[start_mas_services] Port ${port} is occupied by another process (pid=${pid}): ${cmd}"
    echo "[start_mas_services] Please free the port or set MAS_RED_PORT/MAS_BLUE_PORT."
    exit 1
}

cleanup_stale_port "$RED_PORT"
cleanup_stale_port "$BLUE_PORT"

echo "Starting MAS dual-port service..."
echo "Conda Env: $CONDA_ENV"
echo "Host: $HOST"
echo "Ports: red=$RED_PORT blue=$BLUE_PORT"
echo "Prompts File: ${MAS_PROMPTS_FILE}"
echo "Prompt Trace: enabled=${MAS_LOG_PROMPTS} console=${MAS_PROMPT_LOG_CONSOLE} file=${MAS_PROMPT_LOG_FILE}"
echo "Prompt Trace Split: split=${MAS_SPLIT_PROMPT_LOGS} per_run=${MAS_PROMPT_LOG_PER_RUN} run_id=${MAS_RUN_ID}"
echo "Configs Root: $CONFIGS_ROOT"

python "$PYTHON_SCRIPT" \
    --host "$HOST" \
    --red-port "$RED_PORT" \
    --blue-port "$BLUE_PORT" \
    --configs-root "$CONFIGS_ROOT" \
    --log-level "$LOG_LEVEL" &
MAS_PID=$!

echo "MAS service started with PID: $MAS_PID"

cleanup() {
    if [ -n "${MAS_PID:-}" ] && kill -0 "$MAS_PID" 2>/dev/null; then
        echo "Terminating MAS service..."
        kill "$MAS_PID" 2>/dev/null || true
        if kill -0 "$MAS_PID" 2>/dev/null; then
            kill -9 "$MAS_PID" 2>/dev/null || true
        fi
    fi
}

trap cleanup SIGINT SIGTERM EXIT

echo "MAS service is running. Press Ctrl+C to stop."
wait "$MAS_PID" 2>/dev/null
