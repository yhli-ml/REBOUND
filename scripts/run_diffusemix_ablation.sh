#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -jc gs-container_g1.24h
#$ -ac d=nvcr-pytorch-2304

. /fefs/opt/dgx/env_set/nvcr-pytorch-2304-py3.sh

source $HOME/installation_directories_nvcr-pytorch-2304.sh
source $HOME/proxy.sh

cd $HOME/LT_baseline_DA_TL

# =========================================================
# DiffuseMix Ablation: Diagnose WHY augmentation hurts
# =========================================================
# This script runs systematic ablation experiments to isolate
# whether the problem is:
#   A) Domain shift in generated images
#   B) Distribution rebalancing effect
#   C) Hyperparameter mismatch
#   D) Generated data volume (too much noise)
#
# All experiments use ERM (CE loss) for simplicity.
# =========================================================

GPU=${1:-0}
DATA_ROOT="/hss/giil/temp/data"
EPOCHS=200
DATASET="cifar100_lt"
ARCH="resnet32"
DIFFUSEMIX_DIR="./data/diffusemix_cifar100_lt_IF100"

# Common args (matching baseline exactly)
COMMON="--dataset ${DATASET} --data_root ${DATA_ROOT} --arch ${ARCH} \
        --epochs ${EPOCHS} --lr 0.1 --lr_schedule cosine \
        --weight_decay 0.0002 --gpu ${GPU}"

echo "============================================================"
echo " DiffuseMix Ablation Experiments"
echo " Dataset: ${DATASET}, Arch: ${ARCH}, Epochs: ${EPOCHS}"
echo "============================================================"

# ----------------------------------------------------------
# Step 0: Run quality evaluation first (if not done)
# ----------------------------------------------------------
echo ""
echo "[Step 0] Running DiffuseMix quality evaluation..."
if [ ! -f "./eval_diffusemix/summary.json" ]; then
    python evaluate_diffusemix_quality.py \
        --dataset ${DATASET} --data_root ${DATA_ROOT} \
        --diffusemix_dir ${DIFFUSEMIX_DIR} \
        --output_dir ./eval_diffusemix --gpu ${GPU}
else
    echo "  Already done. See ./eval_diffusemix/summary.json"
fi

# ----------------------------------------------------------
# Exp 1: Baseline (no augmentation, cosine schedule)
# Reference point — should match ~39.05%
# ----------------------------------------------------------
echo ""
echo "[Exp 1/8] Baseline ERM (cosine, no augmentation)..."
python main.py ${COMMON} --loss ce \
    --exp_name ablation_baseline

# ----------------------------------------------------------
# Exp 2: DiffuseMix + fix hyperparams (cosine schedule)
# Tests hypothesis: "is it just the lr_schedule difference?"
# ----------------------------------------------------------
echo ""
echo "[Exp 2/8] DiffuseMix + cosine schedule (fair comparison)..."
python main.py ${COMMON} --loss ce \
    --diffusemix_dir ${DIFFUSEMIX_DIR} \
    --exp_name ablation_diffusemix_cosine

# ----------------------------------------------------------
# Exp 3: DiffuseMix + use original cls_num_list
# Tests hypothesis: "does rebalanced distribution hurt the loss?"
# Uses augmented training data but original class counts for
# loss weighting and Many/Medium/Few evaluation
# ----------------------------------------------------------
echo ""
echo "[Exp 3/8] DiffuseMix + original cls_num_list..."
python main.py ${COMMON} --loss ce \
    --diffusemix_dir ${DIFFUSEMIX_DIR} \
    --use_orig_cls_num \
    --exp_name ablation_diffusemix_orig_cls

# ----------------------------------------------------------
# Exp 4: DiffuseMix with only 10% augmented data
# Tests hypothesis: "is the volume of augmented data the problem?"
# ----------------------------------------------------------
echo ""
echo "[Exp 4/8] DiffuseMix 10% ratio..."
python main.py ${COMMON} --loss ce \
    --diffusemix_dir ${DIFFUSEMIX_DIR} \
    --diffusemix_ratio 0.1 \
    --exp_name ablation_diffusemix_ratio10

# ----------------------------------------------------------
# Exp 5: DiffuseMix with 25% augmented data
# ----------------------------------------------------------
echo ""
echo "[Exp 5/8] DiffuseMix 25% ratio..."
python main.py ${COMMON} --loss ce \
    --diffusemix_dir ${DIFFUSEMIX_DIR} \
    --diffusemix_ratio 0.25 \
    --exp_name ablation_diffusemix_ratio25

# ----------------------------------------------------------
# Exp 6: DiffuseMix with 50% augmented data
# ----------------------------------------------------------
echo ""
echo "[Exp 6/8] DiffuseMix 50% ratio..."
python main.py ${COMMON} --loss ce \
    --diffusemix_dir ${DIFFUSEMIX_DIR} \
    --diffusemix_ratio 0.5 \
    --exp_name ablation_diffusemix_ratio50

# ----------------------------------------------------------
# Exp 7: DiffuseMix with loss weight 0.3x for augmented
# Tests hypothesis: "does downweighting augmented samples help?"
# ----------------------------------------------------------
echo ""
echo "[Exp 7/8] DiffuseMix with 0.3x loss weight..."
python main.py ${COMMON} --loss ce \
    --diffusemix_dir ${DIFFUSEMIX_DIR} \
    --diffusemix_weight 0.3 \
    --exp_name ablation_diffusemix_weight03

# ----------------------------------------------------------
# Exp 8: DiffuseMix with loss weight 0.5x for augmented
# ----------------------------------------------------------
echo ""
echo "[Exp 8/8] DiffuseMix with 0.5x loss weight..."
python main.py ${COMMON} --loss ce \
    --diffusemix_dir ${DIFFUSEMIX_DIR} \
    --diffusemix_weight 0.5 \
    --exp_name ablation_diffusemix_weight05

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
echo ""
echo "============================================================"
echo " Ablation Experiments Complete!"
echo "============================================================"
echo ""
echo " Expected interpretation:"
echo "   Exp1 (baseline):        ~39% → reference"
echo "   Exp2 (cosine fix):      if ≈ Exp1, hyperparams were the issue"
echo "                           if << Exp1, domain shift is the issue"
echo "   Exp3 (orig cls_num):    if > Exp2, distribution change hurts loss"
echo "   Exp4-6 (ratio 10-50%):  dose-response curve for data quality"
echo "                           if 10% ≈ baseline, less aug data is better"
echo "   Exp7-8 (loss weight):   if helps, augmented data is noisy but useful"
echo ""
echo " Check results:"
echo "   grep 'Best Acc' output/ablation_*/train.log"
echo "============================================================"
