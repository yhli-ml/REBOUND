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
# iNaturalist 2018: Run baseline methods with ResNet-50
# =========================================================
# Usage: bash scripts/run_inaturalist.sh [GPU_ID] [DATA_ROOT]

GPU=${1:-0}
DATA_ROOT=${2:-"/path/to/iNaturalist18"}
EPOCHS=100
DATASET="inaturalist"
ARCH="resnet50"

echo "==============================================="
echo " iNaturalist 2018 Baselines (${ARCH})"
echo "==============================================="

# 1. ERM
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --weight_decay 5e-4 --gpu ${GPU}

# 2. CB-CE
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss cb_ce --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --weight_decay 5e-4 --gpu ${GPU}

# 3. LDAM-DRW
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ldam --use_norm --drw 60 \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --weight_decay 5e-4 --gpu ${GPU}

# 4. Balanced Softmax
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss balanced_softmax \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --weight_decay 5e-4 --gpu ${GPU}

# 5. cRT
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} \
    --arch ${ARCH} --loss ce --stage2 crt --stage2_epochs 10 \
    --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --weight_decay 5e-4 --gpu ${GPU}

echo "Done!"
