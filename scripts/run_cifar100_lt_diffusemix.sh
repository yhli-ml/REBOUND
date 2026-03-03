#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gs-container_g1.24h
#$ -ac d=nvcr-pytorch-2304

. /fefs/opt/dgx/env_set/nvcr-pytorch-2304-py3.sh

source $HOME/installation_directories_nvcr-pytorch-2304.sh
source $HOME/proxy.sh

cd $HOME/LT_baseline_DA
# =========================================================
# CIFAR-100-LT + DiffuseMix: Two-stage pipeline
# Stage 1: Generate augmented images (offline)
# Stage 2: Train with augmented data
# =========================================================

GPU=${1:-0}
DATA_ROOT="/hss/giil/temp/data"
FRACTAL_DIR="$HOME/diffuseMix/deviantart"
EPOCHS=200
DATASET="cifar100_lt"
ARCH="resnet32"

# ======= Stage 1: Generate DiffuseMix augmented images =======
DIFFUSEMIX_DIR="./data/diffusemix_${DATASET}_IF100"
PROMPTS="sunset,Autumn,watercolor art"
TARGET_NUM=200  # Target samples per class after augmentation

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
echo "[1/6] Training ERM + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 2. CB-CE + DiffuseMix
echo "[2/6] Training CB-CE + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss cb_ce --cb_beta 0.9999 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 3. LDAM-DRW + DiffuseMix
echo "[3/6] Training LDAM-DRW + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ldam --use_norm --drw 160 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 4. Logit Adjustment + DiffuseMix
echo "[4/6] Training Logit Adjustment + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss logit_adjust --la_tau 1.0 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 5. Balanced Softmax + DiffuseMix
echo "[5/6] Training Balanced Softmax + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss balanced_softmax --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 6. cRT + DiffuseMix
echo "[6/6] Training cRT + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --stage2 crt --stage2_epochs 10 \
    --epochs ${EPOCHS} --lr 0.1 --lr_schedule warmup_step --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

echo "==============================================="
echo " All CIFAR-100-LT + DiffuseMix experiments done!"
echo "==============================================="
