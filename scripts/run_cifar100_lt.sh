#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gpu-container_g1.24h
#$ -ac d=nvcr-pytorch-2304

. /fefs/opt/dgx/env_set/nvcr-pytorch-2304-py3.sh

source $HOME/installation_directories_nvcr-pytorch-2304.sh
source $HOME/proxy.sh

cd $HOME/LT_baseline
# =========================================================
# CIFAR-100-LT: Run all baseline methods
# =========================================================
# Usage: bash scripts/run_cifar100_lt.sh [GPU_ID]

GPU=${1:-0}
DATA_ROOT="/hss/giil/temp/data"
EPOCHS=200
DATASET="cifar100_lt"
ARCH="resnet32"

echo "==============================================="
echo " CIFAR-100-LT Baselines (IF=100, ${ARCH})"
echo "==============================================="

# 1. ERM (vanilla cross-entropy)
echo "[1/11] Training ERM..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

# 2. Class-Balanced CE
echo "[2/11] Training CB-CE..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss cb_ce --cb_beta 0.9999 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

# 3. Focal Loss
echo "[3/11] Training Focal Loss..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss focal --focal_gamma 2.0 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

# 4. LDAM
echo "[4/11] Training LDAM..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ldam --use_norm --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

# 5. LDAM-DRW
echo "[5/11] Training LDAM-DRW..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ldam --use_norm --drw 160 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

# 6. Logit Adjustment (training-time)
echo "[6/11] Training Logit Adjustment..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss logit_adjust --la_tau 1.0 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

# 7. Balanced Softmax
echo "[7/11] Training Balanced Softmax..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss balanced_softmax --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

# 8. Seesaw Loss
echo "[8/11] Training Seesaw Loss..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss seesaw --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

# 9. Mixup
echo "[9/11] Training ERM + Mixup..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --mixup --mix_alpha 1.0 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

# 10. CutMix
echo "[10/11] Training ERM + CutMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --cutmix --mix_alpha 1.0 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

# 11. cRT (two-stage: ERM + classifier retraining)
echo "[11/11] Training cRT..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --stage2 crt --stage2_epochs 10 \
    --epochs ${EPOCHS} --lr 0.1 --lr_schedule warmup_step --gpu ${GPU}

echo "==============================================="
echo " All CIFAR-100-LT baselines complete!"
echo "==============================================="
