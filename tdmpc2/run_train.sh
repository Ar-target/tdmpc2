#!/bin/bash

# 遇到错误即刻停止运行
set -e

# ============================================================
# 1. 路径配置
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# work_dir: 存放 checkpoint、eval 视频、训练曲线 CSV
WORK_DIR="${SCRIPT_DIR}/runs"

# data_dir: 存放 replay buffer 数据
DATA_DIR="${SCRIPT_DIR}/data"

# 自动创建目录（如不存在）
mkdir -p "${WORK_DIR}"
mkdir -p "${DATA_DIR}"

# ============================================================
# 2. 训练参数
# ============================================================
TASK="beamng-utah"
EXP_NAME="utah_avoidance_v1"
SEED=42
MODEL_SIZE=5

echo "=========================================================="
echo "🚀 开始训练 TD-MPC2"
echo "📍 任务:      ${TASK}"
echo "🏷️  实验名:    ${EXP_NAME}"
echo "🧠 模型大小:  ${MODEL_SIZE}M 参数"
echo "📁 输出目录:  ${WORK_DIR}"
echo "💾 数据目录:  ${DATA_DIR}"
echo "=========================================================="

# ============================================================
# 3. 运行训练
# ============================================================
python train.py \
    task=${TASK} \
    exp_name=${EXP_NAME} \
    seed=${SEED} \
    model_size=${MODEL_SIZE} \
    work_dir=${WORK_DIR} \
    data_dir=${DATA_DIR} \
    wandb_project=tdmpc2_beamng \
    wandb_entity=models-university-system-of-georgia

echo "✅ 训练进程已结束或被手动终止。"