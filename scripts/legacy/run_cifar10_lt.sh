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
# CIFAR-10-LT: Run all baseline methods
# =========================================================
# Usage: bash scripts/run_cifar10_lt.sh [GPU_ID]

GPU=${1:-0}
DATA_ROOT="/hss/giil/temp/data"
EPOCHS=200
DATASET="cifar10_lt"
ARCH="resnet32"

echo "==============================================="
echo " CIFAR-10-LT Baselines (IF=100, ${ARCH})"
echo "==============================================="

# 1. ERM
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule cosine --gpu ${GPU}

# 2. CB-CE
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss cb_ce --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule cosine --gpu ${GPU}

# 3. Focal Loss
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss focal --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule cosine --gpu ${GPU}

# 4. LDAM-DRW
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ldam --use_norm --drw 160 --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule cosine --gpu ${GPU}

# 5. Logit Adjustment
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss logit_adjust --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule cosine --gpu ${GPU}

# 6. Balanced Softmax
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss balanced_softmax --epochs ${EPOCHS} \
    --lr 0.1 --lr_schedule cosine --gpu ${GPU}

echo "Done!"
