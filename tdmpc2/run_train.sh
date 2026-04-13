#!/bin/bash

# 遇到错误即刻停止运行
set -e

# 2. 定义关键训练参数
TASK="beamng-utah"
EXP_NAME="utah_avoidance_v1"
SEED=42
MODEL_SIZE=5  # TD-MPC2 的模型大小 (可用选项通常为 1, 5, 19, 48，代表参数量百万级别，5M 是默认平衡点)

echo "=========================================================="
echo "🚀 开始训练 TD-MPC2"
echo "📍 任务: ${TASK}"
echo "🏷️ 实验名: ${EXP_NAME}"
echo "🧠 模型大小: ${MODEL_SIZE}M 参数"
echo "=========================================================="

# 3. 运行训练脚本
# 这里通过命令行参数补全 config.yaml 中未定义的 "???" 变量，避免报错
python train.py \
    task=${TASK} \
    exp_name=${EXP_NAME} \
    seed=${SEED} \
    model_size=${MODEL_SIZE}

echo "✅ 训练进程已结束或被手动终止。"