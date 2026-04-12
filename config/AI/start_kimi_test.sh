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

python ~/robotproject_ws/src/robot_vs/scripts/AI/kimi_test.py