#!/bin/bash

# 创建一个新的 tmux 会话，名为 soccer_exam，如果已存在则附加
SESSION="vqa-clg"
tmux has-session -t $SESSION 2>/dev/null
if [ $? != 0 ]; then
    tmux new-session -d -s $SESSION
fi

# 在 tmux 会话中激活指定conda环境、cd到指定目录、设置API Key，并运行Python脚本
tmux send-keys -t $SESSION "
conda activate sn
cd /home/zhaosiyao/SoccerAgent/baseline
export OPENROUTER_API_KEY=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
python baseline.py \
    --input_file /data/zhaosiyao/SoccerNet/SoccerNet-SN-VQA-2026/challenge/challenge.json \
    --materials_folder /data/zhaosiyao/SoccerNet/SoccerNet-SN-VQA-2026/challenge \
    --output_file metadata_challenge_$(date +%Y%m%d_%H%M%S).json \
    --model google/gemini-3.1-flash-lite-preview
" C-m

# 可选：自动附加到tmux会话
tmux attach-session -t $SESSION