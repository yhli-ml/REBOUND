#!/bin/bash
#SBATCH --partition=lab-chandar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mem=30G
#SBATCH --time=2-0:00:00
#SBATCH -o /home/mila/j/jerry.huang/code_yuhangli/REBOUND/scripts/slurm-cifar100_lt_diffusemix-%j.out

set -euo pipefail

module --force purge
module --quiet load anaconda/3

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pt_env}"
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/mila/j/jerry.huang/code_yuhangli/REBOUND}"
cd "${PROJECT_ROOT}"
# =========================================================
# CIFAR-100-LT + DiffuseMix reference baseline: Two-stage pipeline
# Stage 1: Generate augmented images (offline)
# Stage 2: Train with augmented data
# =========================================================

GPU=${1:-0}
DATA_ROOT="./data"
FRACTAL_DIR="./deviantart"
EPOCHS=200
DATASET="cifar100_lt"
ARCH="resnet32"

# ======= Stage 1: Generate DiffuseMix augmented images =======
DIFFUSEMIX_DIR="./data/diffusemix_${DATASET}_IF100"
PROMPTS="sunset,Autumn,watercolor art"
AUG_PER_IMG=1  # Per-image DiffuseMix baseline, aligned with Head2Tail plan

echo "==============================================="
echo " Stage 1: DiffuseMix Reference Baseline"
echo " Dataset: ${DATASET}, Plan: per_image (m=${AUG_PER_IMG})"
echo "==============================================="

if [ ! -f "${DIFFUSEMIX_DIR}/metadata.json" ]; then
    python generate_diffusemix.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --imb_factor 0.01 \
        --fractal_dir ${FRACTAL_DIR} \
        --prompts "${PROMPTS}" \
        --plan_mode per_image \
        --aug_per_image ${AUG_PER_IMG} \
        --output_dir ${DIFFUSEMIX_DIR} \
        --gen_size 256 \
        --save_size 32 \
        --device cuda \
        --seed 42
    echo "Stage 1 complete."
else
    echo "Augmented data already exists at ${DIFFUSEMIX_DIR}, skipping generation."
fi

# ======= Stage 2: Train with DiffuseMix augmented data =======
echo ""
echo "==============================================="
echo " Stage 2: Training with DiffuseMix reference baseline"
echo "==============================================="

# 1. ERM + DiffuseMix
echo "[1/6] Training ERM + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# # 2. CB-CE + DiffuseMix
# echo "[2/6] Training CB-CE + DiffuseMix..."
# python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
#     --arch ${ARCH} --loss cb_ce --cb_beta 0.9999 --epochs ${EPOCHS} \
#     --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
#     --diffusemix_dir ${DIFFUSEMIX_DIR}

# # 3. LDAM-DRW + DiffuseMix
# echo "[3/6] Training LDAM-DRW + DiffuseMix..."
# python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
#     --arch ${ARCH} --loss ldam --use_norm --drw 160 --epochs ${EPOCHS} \
#     --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
#     --diffusemix_dir ${DIFFUSEMIX_DIR}

# # 4. Logit Adjustment + DiffuseMix
# echo "[4/6] Training Logit Adjustment + DiffuseMix..."
# python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
#     --arch ${ARCH} --loss logit_adjust --la_tau 1.0 --epochs ${EPOCHS} \
#     --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
#     --diffusemix_dir ${DIFFUSEMIX_DIR}

# # 5. Balanced Softmax + DiffuseMix
# echo "[5/6] Training Balanced Softmax + DiffuseMix..."
# python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
#     --arch ${ARCH} --loss balanced_softmax --epochs ${EPOCHS} \
#     --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
#     --diffusemix_dir ${DIFFUSEMIX_DIR}

# # 6. cRT + DiffuseMix
# echo "[6/6] Training cRT + DiffuseMix..."
# python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
#     --arch ${ARCH} --loss ce --stage2 crt --stage2_epochs 10 \
#     --epochs ${EPOCHS} --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
#     --diffusemix_dir ${DIFFUSEMIX_DIR}

echo "==============================================="
echo " All CIFAR-100-LT + DiffuseMix experiments done!"
echo "==============================================="
