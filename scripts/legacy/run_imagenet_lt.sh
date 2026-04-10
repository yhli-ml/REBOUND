#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gs-container_g1.24h
#$ -ac d=nvcr-pytorch-2304

. /fefs/opt/dgx/env_set/nvcr-pytorch-2304-py3.sh

source $HOME/installation_directories_nvcr-pytorch-2304.sh
source $HOME/proxy.sh

cd $HOME/LT_baseline
# =========================================================
# ImageNet-LT: Run baseline methods with ResNet-50
# =========================================================
# Usage: bash scripts/run_imagenet_lt.sh [GPU_ID] [DATA_ROOT]

GPU=${1:-0}
DATA_ROOT="$HOME/LT_baseline/data/ImageNet_LT"
IMG_ROOT="/hss/giil/temp/data/imagenet"
EPOCHS=100
DATASET="imagenet_lt"
ARCH="resnet50"

echo "==============================================="
echo " ImageNet-LT Baselines (${ARCH})"
echo "==============================================="

# 1. ERM
echo "[1/8] Training ERM..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss ce --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU}

# 2. CB-CE
echo "[2/8] Training CB-CE..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss cb_ce --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU}

# 3. LDAM-DRW
echo "[3/8] Training LDAM-DRW..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss ldam --use_norm --drw 60 \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU}

# 4. Logit Adjustment
echo "[4/8] Training Logit Adjustment..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss logit_adjust --la_tau 1.0 \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU}

# 5. Balanced Softmax
echo "[5/8] Training Balanced Softmax..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss balanced_softmax \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU}

# 6. Seesaw Loss
echo "[6/8] Training Seesaw Loss..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss seesaw \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU}

# 7. cRT
echo "[7/8] Training cRT..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss ce --stage2 crt --stage2_epochs 10 \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU}

# 8. ERM + tau-norm
echo "[8/8] Training ERM + tau-norm..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss ce --stage2 tau_norm --tau 1.0 \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU}

echo "==============================================="
echo " All ImageNet-LT baselines complete!"
echo "==============================================="
