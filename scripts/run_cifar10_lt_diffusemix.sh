#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gpu-container_g1.24h
#$ -ac d=nvcr-pytorch-2304

. /fefs/opt/dgx/env_set/nvcr-pytorch-2304-py3.sh

source $HOME/installation_directories_nvcr-pytorch-2304.sh
source $HOME/proxy.sh

cd $HOME/LT_baseline_DA
# =========================================================
# CIFAR-10-LT + DiffuseMix: Two-stage pipeline
# =========================================================

GPU=${1:-0}
DATA_ROOT="/hss/giil/temp/data"
FRACTAL_DIR="$HOME/diffuseMix/deviantart"
EPOCHS=200
DATASET="cifar10_lt"
ARCH="resnet32"

# ======= Stage 1: Generate DiffuseMix augmented images =======
DIFFUSEMIX_DIR="./data/diffusemix_${DATASET}_IF100"
PROMPTS="sunset,Autumn,watercolor art"
TARGET_NUM=2000  # Target for CIFAR-10 (max class ~5000)

echo "==============================================="
echo " Stage 1: DiffuseMix Offline Augmentation"
echo " Dataset: ${DATASET}, Target: ${TARGET_NUM}/class"
echo "==============================================="

if [ ! -f "${DIFFUSEMIX_DIR}/metadata.json" ]; then
    python generate_diffusemix.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --imb_factor 0.01 \
        --fractal_dir ${FRACTAL_DIR} \
        --prompts "${PROMPTS}" \
        --target_num ${TARGET_NUM} \
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
echo " Stage 2: Training with DiffuseMix"
echo "==============================================="

# 1. ERM + DiffuseMix
echo "[1/4] Training ERM + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 2. LDAM-DRW + DiffuseMix
echo "[2/4] Training LDAM-DRW + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ldam --use_norm --drw 160 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 3. Balanced Softmax + DiffuseMix
echo "[3/4] Training Balanced Softmax + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss balanced_softmax --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 4. Logit Adjustment + DiffuseMix
echo "[4/4] Training Logit Adjustment + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss logit_adjust --la_tau 1.0 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

echo "==============================================="
echo " All CIFAR-10-LT + DiffuseMix experiments done!"
echo "==============================================="
