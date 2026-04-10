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

# ===============================================================
# Head2Tail Ablation Study
# ===============================================================
#
# These ablations intentionally use the legacy target-mode generator to compare
# against earlier runs.
#
# Ablation 1: Head selection strategy (nearest / random / farthest)
# Ablation 2: SDEdit strength (0.4 / 0.5 / 0.6 / 0.7 / 0.8)
# Ablation 3: Top-K nearest heads (1 / 3 / 5)
#
# Each ablation generates images → trains classifier → evaluates
# ===============================================================

DATASET="cifar100_lt"
DATA_ROOT="./data"
IMB_FACTOR=0.01
SEED=42
TARGET_NUM=500

# ---------------------------------------------------------------
# Ablation 1: Head class selection strategy
# ---------------------------------------------------------------
echo "========================================="
echo "Ablation 1: Head Selection Strategy"
echo "========================================="

for STRATEGY in nearest random farthest; do
    echo ""
    echo "--- Strategy: ${STRATEGY} ---"
    
    OUTPUT_DIR="./data/head2tail_ablation_selection_${STRATEGY}"
    
    python generate_head2tail.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --output_dir ${OUTPUT_DIR} \
        --top_k 3 \
        --head_selection ${STRATEGY} \
        --strength 0.6 \
        --guidance_scale 7.5 \
        --gen_size 512 \
        --save_size 32 \
        --plan_mode target \
        --target_num ${TARGET_NUM} \
        --seed ${SEED}
    
    python main.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --arch resnet32 \
        --loss ce \
        --epochs 200 \
        --lr 0.1 \
        --lr_schedule warmup_step \
        --batch_size 128 \
        --diffusemix_dir ${OUTPUT_DIR} \
        --use_orig_cls_num \
        --output_dir ./output \
        --exp_name ablation_selection_${STRATEGY} \
        --seed ${SEED}
done

# ---------------------------------------------------------------
# Ablation 2: SDEdit strength
# ---------------------------------------------------------------
echo ""
echo "========================================="
echo "Ablation 2: SDEdit Strength"
echo "========================================="

for STRENGTH in 0.4 0.5 0.6 0.7 0.8; do
    echo ""
    echo "--- Strength: ${STRENGTH} ---"
    
    OUTPUT_DIR="./data/head2tail_ablation_strength_${STRENGTH}"
    
    python generate_head2tail.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --output_dir ${OUTPUT_DIR} \
        --top_k 3 \
        --head_selection nearest \
        --strength ${STRENGTH} \
        --guidance_scale 7.5 \
        --gen_size 512 \
        --save_size 32 \
        --plan_mode target \
        --target_num ${TARGET_NUM} \
        --seed ${SEED}
    
    python main.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --arch resnet32 \
        --loss ce \
        --epochs 200 \
        --lr 0.1 \
        --lr_schedule warmup_step \
        --batch_size 128 \
        --diffusemix_dir ${OUTPUT_DIR} \
        --use_orig_cls_num \
        --output_dir ./output \
        --exp_name ablation_strength_${STRENGTH} \
        --seed ${SEED}
done

# ---------------------------------------------------------------
# Ablation 3: Top-K nearest heads
# ---------------------------------------------------------------
echo ""
echo "========================================="
echo "Ablation 3: Top-K"
echo "========================================="

for K in 1 3 5; do
    echo ""
    echo "--- Top-K: ${K} ---"
    
    OUTPUT_DIR="./data/head2tail_ablation_topk_${K}"
    
    python generate_head2tail.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --output_dir ${OUTPUT_DIR} \
        --top_k ${K} \
        --head_selection nearest \
        --strength 0.6 \
        --guidance_scale 7.5 \
        --gen_size 512 \
        --save_size 32 \
        --plan_mode target \
        --target_num ${TARGET_NUM} \
        --seed ${SEED}
    
    python main.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --arch resnet32 \
        --loss ce \
        --epochs 200 \
        --lr 0.1 \
        --lr_schedule warmup_step \
        --batch_size 128 \
        --diffusemix_dir ${OUTPUT_DIR} \
        --use_orig_cls_num \
        --output_dir ./output \
        --exp_name ablation_topk_${K} \
        --seed ${SEED}
done

echo ""
echo "All ablation experiments complete."
