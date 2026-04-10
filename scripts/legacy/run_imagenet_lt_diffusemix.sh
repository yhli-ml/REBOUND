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
# ImageNet-LT + DiffuseMix reference baseline: Two-stage pipeline
# =========================================================

GPU=${1:-0}
DATA_ROOT="$HOME/LT_baseline_DA/data/ImageNet_LT"
IMG_ROOT="/hss/giil/temp/data/imagenet"
FRACTAL_DIR="$HOME/diffuseMix/deviantart"
EPOCHS=100
DATASET="imagenet_lt"
ARCH="resnet50"

# ======= Stage 1: Generate DiffuseMix augmented images =======
DIFFUSEMIX_DIR="./data/diffusemix_${DATASET}"
PROMPTS="sunset,Autumn"
AUG_PER_IMG=1  # Per-image DiffuseMix baseline, aligned with Head2Tail plan

echo "==============================================="
echo " Stage 1: DiffuseMix Reference Baseline"
echo " Dataset: ${DATASET}, Plan: per_image (m=${AUG_PER_IMG})"
echo "==============================================="

if [ ! -f "${DIFFUSEMIX_DIR}/metadata.json" ]; then
    python generate_diffusemix.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --img_root ${IMG_ROOT} \
        --fractal_dir ${FRACTAL_DIR} \
        --prompts "${PROMPTS}" \
        --plan_mode per_image \
        --aug_per_image ${AUG_PER_IMG} \
        --output_dir ${DIFFUSEMIX_DIR} \
        --gen_size 256 \
        --save_size 0 \
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
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss ce --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 2. CB-CE + DiffuseMix
echo "[2/6] Training CB-CE + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss cb_ce --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 3. LDAM-DRW + DiffuseMix
echo "[3/6] Training LDAM-DRW + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss ldam --use_norm --drw 60 \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 4. Balanced Softmax + DiffuseMix
echo "[4/6] Training Balanced Softmax + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss balanced_softmax \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 5. Logit Adjustment + DiffuseMix
echo "[5/6] Training Logit Adjustment + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss logit_adjust --la_tau 1.0 \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# 6. cRT + DiffuseMix
echo "[6/6] Training cRT + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss ce --stage2 crt --stage2_epochs 10 \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

echo "==============================================="
echo " All ImageNet-LT + DiffuseMix experiments done!"
echo "==============================================="
