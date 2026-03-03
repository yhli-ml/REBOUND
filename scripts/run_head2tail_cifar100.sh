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
# Head-to-Tail Augmentation Pipeline for CIFAR-100-LT
# ===============================================================
#
# Full pipeline:
#   Step 0 (Optional): LoRA fine-tune SD on CIFAR-100-LT
#   Step 1: Generate head→tail augmented images
#   Step 2: Train classifier with augmented data
#
# Ablation experiments (uncomment desired configuration):
#   - Head selection: nearest / random / farthest / all
#   - Strength: 0.4 / 0.5 / 0.6 / 0.7 / 0.8
#   - Top-K: 1 / 3 / 5 / all
# ===============================================================

# Common settings
DATASET="cifar100_lt"
DATA_ROOT="./data"
IMB_FACTOR=0.01
SEED=42

# ---------------------------------------------------------------
# Step 0 (Optional): LoRA Fine-tuning
# ---------------------------------------------------------------
# Uncomment to fine-tune SD on CIFAR-100-LT first
# This helps the diffusion model understand the CIFAR domain
if [ ! -d "./lora_weights/${DATASET}_IF100" ]; then # Only run if LoRA weights don't already exist
    python -m augment.head2tail_lora_finetune \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --output_dir ./lora_weights/${DATASET}_IF100 \
        --resolution 512 \
        --lora_rank 4 \
        --train_steps 2000 \
        --batch_size 4 \
        --lr 1e-4 \
        --seed ${SEED}
else
    echo "LoRA weights already exist for ${DATASET}, skipping fine-tuning."
fi

LORA_WEIGHTS="./lora_weights/${DATASET}_IF100/final"

# ---------------------------------------------------------------
# Step 1: Generate Head-to-Tail augmented images
# ---------------------------------------------------------------

# Generation parameters
TOP_K=3
HEAD_THRESH=100
TAIL_THRESH=20
FEAT_SRC="clip_image"
HEAD_SEL="nearest"
SAMPLE_SEL="random"
MODEL_ID="runwayml/stable-diffusion-v1-5"
PIPE_TYPE="img2img"
STRENGTH=0.6
GUIDANCE=7.5
STEPS=50
GEN_SIZE=512
SAVE_SIZE=32
N_PROMPTS=5
PLAN_MODE="per_image"
AUG_PER_IMG=1

# Training parameters
ARCH="resnet32"
LOSS="ce"
EPOCHS=200
LR=0.1
LR_SCHED="cosine"
BATCH_SIZE=128
DMIX_RATIO=1.0
WD=0.0002

# Build a descriptive tag with all key parameters
# Format: h2t_{dataset}_IF{imb}_{feat}_{sel}_k{K}_s{str}_g{guid}_st{steps}_{pipe}_{plan}_m{aug}_p{nprompt}_seed{s}
GEN_TAG="h2t_${DATASET}_IF${IMB_FACTOR}"
GEN_TAG="${GEN_TAG}_${FEAT_SRC}_${HEAD_SEL}_${SAMPLE_SEL}"
GEN_TAG="${GEN_TAG}_k${TOP_K}_ht${HEAD_THRESH}_tt${TAIL_THRESH}"
GEN_TAG="${GEN_TAG}_s${STRENGTH}_g${GUIDANCE}_st${STEPS}"
GEN_TAG="${GEN_TAG}_${PIPE_TYPE}_${PLAN_MODE}_m${AUG_PER_IMG}"
GEN_TAG="${GEN_TAG}_p${N_PROMPTS}_seed${SEED}"
# Add LoRA indicator
if [ -n "${LORA_WEIGHTS}" ]; then
    GEN_TAG="${GEN_TAG}_lora"
fi

# If tag is too long (>120 chars), append a short hash and truncate
if [ ${#GEN_TAG} -gt 120 ]; then
    HASH=$(echo "${GEN_TAG}" | md5sum | cut -c1-8)
    GEN_TAG="${GEN_TAG:0:100}_${HASH}"
fi

OUTPUT_DIR="./data/${GEN_TAG}"

# Training tag
TRAIN_TAG="${GEN_TAG}_${ARCH}_${LOSS}_ep${EPOCHS}_lr${LR}_${LR_SCHED}_bs${BATCH_SIZE}_wd${WD}_dr${DMIX_RATIO}"
if [ ${#TRAIN_TAG} -gt 200 ]; then
    HASH2=$(echo "${TRAIN_TAG}" | md5sum | cut -c1-8)
    TRAIN_TAG="${TRAIN_TAG:0:180}_${HASH2}"
fi
EXP_NAME="${TRAIN_TAG}"

echo "Output dir: ${OUTPUT_DIR}"
echo "Exp name:   ${EXP_NAME}"

python generate_head2tail.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --imb_factor ${IMB_FACTOR} \
    --output_dir ${OUTPUT_DIR} \
    --top_k ${TOP_K} \
    --head_threshold ${HEAD_THRESH} \
    --tail_threshold ${TAIL_THRESH} \
    --feature_source ${FEAT_SRC} \
    --head_selection ${HEAD_SEL} \
    --sample_selection ${SAMPLE_SEL} \
    --model_id ${MODEL_ID} \
    --pipeline_type ${PIPE_TYPE} \
    --lora_weights "${LORA_WEIGHTS}" \
    --strength ${STRENGTH} \
    --guidance_scale ${GUIDANCE} \
    --num_inference_steps ${STEPS} \
    --gen_size ${GEN_SIZE} \
    --save_size ${SAVE_SIZE} \
    --n_prompts_per_class ${N_PROMPTS} \
    --plan_mode ${PLAN_MODE} \
    --aug_per_image ${AUG_PER_IMG} \
    --seed ${SEED}

echo "Generation complete. Output: ${OUTPUT_DIR}"

# ---------------------------------------------------------------
# Step 2: Train classifier with Head-to-Tail augmented data
# ---------------------------------------------------------------
python main.py \
    --dataset ${DATASET} \
    --data_root ${DATA_ROOT} \
    --imb_factor ${IMB_FACTOR} \
    --arch ${ARCH} \
    --loss ${LOSS} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --lr_schedule ${LR_SCHED} \
    --batch_size ${BATCH_SIZE} \
    --weight_decay ${WD} \
    --diffusemix_dir ${OUTPUT_DIR} \
    --diffusemix_ratio ${DMIX_RATIO} \
    --use_orig_cls_num \
    --output_dir ./output \
    --exp_name ${EXP_NAME} \
    --seed ${SEED}

echo "Training complete."
