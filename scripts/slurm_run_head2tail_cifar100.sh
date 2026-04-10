#!/bin/bash
#SBATCH --partition=lab-chandar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mem=30G
#SBATCH --time=2-0:00:00
#SBATCH -o /home/mila/j/jerry.huang/code_yuhangli/REBOUND/scripts/slurm-cifar100_lt_head2tail-%j.out

set -euo pipefail

module --force purge
module --quiet load anaconda/3

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pt_env}"
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/mila/j/jerry.huang/code_yuhangli/REBOUND}"
cd "${PROJECT_ROOT}"

# ===============================================================
# Head2Tail proposed method for CIFAR-100-LT
# ===============================================================
#
# Full pipeline:
#   Step 0 (Optional): LoRA fine-tune SD on CIFAR-100-LT
#   Step 1: Train baseline classifier on original images only
#   Step 2: Generate head→tail augmented images (skip if already present)
#   Step 3: Train classifier with original + Head2Tail augmented data
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
LORA_NAME="${LORA_NAME:-}"

# ---------------------------------------------------------------
# Step 1: LoRA Fine-tuning
# ---------------------------------------------------------------
# Uncomment to fine-tune SD on CIFAR-100-LT first
# This helps the diffusion model understand the CIFAR domain
LORA_DIR="./lora_weights/${DATASET}_IF100"
if [ -n "${LORA_NAME}" ]; then
    LORA_DIR="${LORA_DIR}_${LORA_NAME}"
fi
if [ ! -d "${LORA_DIR}/final" ]; then # Only run if LoRA weights don't already exist
    python -m augment.head2tail_lora_finetune \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --output_dir ${LORA_DIR} \
        --resolution 512 \
        --lora_rank 4 \
        --train_steps 2000 \
        --batch_size 4 \
        --lr 1e-4 \
        --sampling_strategy uniform \
        --caption_mode templated \
        --seed ${SEED}
else
    echo "LoRA weights already exist for ${DATASET}, skipping fine-tuning."
fi


LORA_WEIGHTS="${LORA_DIR}/final"

# ---------------------------------------------------------------
# Step 2: Generate Head-to-Tail augmented images
# ---------------------------------------------------------------

# Generation parameters
TOP_K=3
HEAD_THRESH=100
TAIL_THRESH=20
FEAT_SRC="clip_image" # important
HEAD_SEL="nearest" # important
SAMPLE_SEL="random" # important
MODEL_ID="runwayml/stable-diffusion-v1-5"
PIPE_TYPE="img2img"
STRENGTH=0.6
GUIDANCE=7.5
STEPS=50
GEN_SIZE=512
SAVE_SIZE=32
N_PROMPTS=5 # important
PLAN_MODE="per_image"
PER_IMAGE_SCOPE="all" # important

AUG_PER_IMG=1 # important

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
GEN_TAG="${GEN_TAG}_${PIPE_TYPE}_${PLAN_MODE}_${PER_IMAGE_SCOPE}_m${AUG_PER_IMG}"
GEN_TAG="${GEN_TAG}_p${N_PROMPTS}_seed${SEED}"
# Add LoRA indicator
if [ -n "${LORA_WEIGHTS}" ]; then
    if [ -n "${LORA_NAME}" ]; then
        GEN_TAG="${GEN_TAG}_${LORA_NAME}"
    else
        GEN_TAG="${GEN_TAG}_lora"
    fi
fi

# If tag is too long (>120 chars), append a short hash and truncate
if [ ${#GEN_TAG} -gt 120 ]; then
    HASH=$(echo "${GEN_TAG}" | md5sum | cut -c1-8)
    GEN_TAG="${GEN_TAG:0:100}_${HASH}"
fi

OUTPUT_DIR="./data/${GEN_TAG}"

# Training tags
BASELINE_TAG="baseline_${DATASET}_IF${IMB_FACTOR}_${ARCH}_${LOSS}_ep${EPOCHS}_lr${LR}_${LR_SCHED}_bs${BATCH_SIZE}_wd${WD}_seed${SEED}"
if [ ${#BASELINE_TAG} -gt 200 ]; then
    HASH_BASE=$(echo "${BASELINE_TAG}" | md5sum | cut -c1-8)
    BASELINE_TAG="${BASELINE_TAG:0:180}_${HASH_BASE}"
fi
H2T_TAG="${GEN_TAG}_${ARCH}_${LOSS}_ep${EPOCHS}_lr${LR}_${LR_SCHED}_bs${BATCH_SIZE}_wd${WD}_dr${DMIX_RATIO}"
if [ ${#H2T_TAG} -gt 200 ]; then
    HASH2=$(echo "${H2T_TAG}" | md5sum | cut -c1-8)
    H2T_TAG="${H2T_TAG:0:180}_${HASH2}"
fi

echo "Output dir: ${OUTPUT_DIR}"
echo "LoRA dir: ${LORA_DIR}"
echo "LoRA name: ${LORA_NAME}"
echo "Baseline exp: ${BASELINE_TAG}"
echo "Head2Tail exp: ${H2T_TAG}"

# ---------------------------------------------------------------
# Step 1: Train baseline classifier on original images only
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
    --output_dir ./output \
    --exp_name ${BASELINE_TAG} \
    --seed ${SEED}

echo "Baseline training complete."

# ---------------------------------------------------------------
# Step 2: Generate Head-to-Tail augmented images
# ---------------------------------------------------------------
if [ -f "${OUTPUT_DIR}/metadata.json" ]; then
    echo "Found existing Head2Tail metadata at ${OUTPUT_DIR}/metadata.json, skipping generation."
else
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
        --per_image_scope ${PER_IMAGE_SCOPE} \
        --aug_per_image ${AUG_PER_IMG} \
        --seed ${SEED}

    echo "Generation complete. Output: ${OUTPUT_DIR}"
fi

# ---------------------------------------------------------------
# Step 3: Train classifier with Head-to-Tail augmented data
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
    --exp_name ${H2T_TAG} \
    --seed ${SEED}

echo "Head2Tail training complete."
