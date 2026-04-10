#!/bin/bash
#SBATCH --partition=lab-chandar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mem=80G
#SBATCH --time=2-0:00:00
#SBATCH -o /home/mila/j/jerry.huang/code_yuhangli/REBOUND/scripts/slurm-imagenet_lt_head2tail_controlled-%j.out

set -euo pipefail

module --force purge
module --quiet load anaconda/3

CONDA_ENV_NAME="${CONDA_ENV_NAME:-pt_env}"
export PS1="${PS1:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/mila/j/jerry.huang/code_yuhangli/REBOUND}"
cd "${PROJECT_ROOT}"

IMAGENET_ROOT="${SLURM_TMPDIR}/imagenet"
mkdir -p "${IMAGENET_ROOT}/train" "${IMAGENET_ROOT}/val"

cp /network/datasets/imagenet/ILSVRC2012_img_train.tar "${SLURM_TMPDIR}/"
tar -xf "${SLURM_TMPDIR}/ILSVRC2012_img_train.tar" -C "${IMAGENET_ROOT}/train"
cd "${IMAGENET_ROOT}/train"
find . -name "*.tar" | while read -r NAME ; do
  mkdir -p "${NAME%.tar}"
  tar -xf "${NAME}" -C "${NAME%.tar}"
  rm -f "${NAME}"
done

cp /network/datasets/imagenet/ILSVRC2012_img_val.tar "${SLURM_TMPDIR}/"
tar -xf "${SLURM_TMPDIR}/ILSVRC2012_img_val.tar" -C "${IMAGENET_ROOT}/val"
cd "${IMAGENET_ROOT}/val"
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

cd "${PROJECT_ROOT}"

# ===============================================================
# Controlled Head2Tail for ImageNet-LT
# ===============================================================
# Use H2T_VARIANT=random|nearest|nearest_filter|nearest_neg_filter.
# Full ImageNet uniformization can be very large, so TARGET_NUM defaults to 100.
# Set TARGET_NUM=-1 and increase MAX_AUG_PER_CLASS only when you explicitly want
# full max-class uniformization and have enough generation budget.
# ===============================================================

DATASET="imagenet_lt"
DATA_ROOT="${DATA_ROOT:-./data/ImageNet_LT}"
IMG_ROOT="${IMG_ROOT:-${IMAGENET_ROOT}}"
IMB_FACTOR="${IMB_FACTOR:-0.01}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
H2T_VARIANT="${H2T_VARIANT:-nearest_neg_filter}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
RUN_BASELINE="${RUN_BASELINE:-0}"
RUN_TRAIN="${RUN_TRAIN:-1}"

ENABLE_LORA="${ENABLE_LORA:-0}"
LORA_NAME="${LORA_NAME:-controlled_uniform}"
LORA_DIR="./lora_weights/${DATASET}_${LORA_NAME}"
LORA_WEIGHTS=""
if [ "${ENABLE_LORA}" = "1" ]; then
    if [ ! -d "${LORA_DIR}/final" ]; then
        python -m augment.head2tail_lora_finetune \
            --dataset ${DATASET} \
            --data_root ${DATA_ROOT} \
            --img_root ${IMG_ROOT} \
            --imb_factor ${IMB_FACTOR} \
            --output_dir ${LORA_DIR} \
            --resolution 512 \
            --lora_rank 4 \
            --train_steps "${LORA_TRAIN_STEPS:-2000}" \
            --batch_size 4 \
            --lr 1e-4 \
            --sampling_strategy uniform \
            --caption_mode templated \
            --seed ${SEED}
    else
        echo "LoRA weights already exist at ${LORA_DIR}/final, skipping fine-tuning."
    fi
    LORA_WEIGHTS="${LORA_DIR}/final"
fi

TOP_K="${TOP_K:-3}"
HEAD_THRESH="${HEAD_THRESH:-100}"
TAIL_THRESH="${TAIL_THRESH:-20}"
FEAT_SRC="${FEAT_SRC:-clip_image}"
HEAD_SEL="${HEAD_SEL:-nearest}"
MODEL_ID="${MODEL_ID:-runwayml/stable-diffusion-v1-5}"
PIPE_TYPE="${PIPE_TYPE:-img2img}"
STRENGTH="${STRENGTH:-0.6}"
GUIDANCE="${GUIDANCE:-7.5}"
STEPS="${STEPS:-50}"
GEN_SIZE="${GEN_SIZE:-512}"
SAVE_SIZE="${SAVE_SIZE:-256}"
N_PROMPTS="${N_PROMPTS:-5}"

PLAN_MODE="${PLAN_MODE:-uniformize}"
PER_IMAGE_SCOPE="${PER_IMAGE_SCOPE:-medium_tail}"
TARGET_NUM="${TARGET_NUM:-100}"
MAX_AUG_PER_CLASS="${MAX_AUG_PER_CLASS:-100}"
MAX_ATTEMPT_MULTIPLIER="${MAX_ATTEMPT_MULTIPLIER:-4.0}"

TARGET_TEXT_THRESHOLD="${TARGET_TEXT_THRESHOLD:--1.0}"
PROTOTYPE_MARGIN_THRESHOLD="${PROTOTYPE_MARGIN_THRESHOLD:-0.0}"
KEEP_TOP_K_PER_CLASS="${KEEP_TOP_K_PER_CLASS:-0}"
KEEP_RATIO="${KEEP_RATIO:-1.0}"

case "${H2T_VARIANT}" in
    random)
        SAMPLE_SEL="random"
        ENABLE_FILTER="0"
        SOURCE_NEGATIVE_PROMPT="0"
        ;;
    nearest)
        SAMPLE_SEL="nearest"
        ENABLE_FILTER="0"
        SOURCE_NEGATIVE_PROMPT="0"
        ;;
    nearest_filter)
        SAMPLE_SEL="nearest"
        ENABLE_FILTER="1"
        SOURCE_NEGATIVE_PROMPT="0"
        ;;
    nearest_neg_filter)
        SAMPLE_SEL="nearest"
        ENABLE_FILTER="1"
        SOURCE_NEGATIVE_PROMPT="1"
        ;;
    *)
        echo "Unknown H2T_VARIANT=${H2T_VARIANT}" >&2
        exit 2
        ;;
esac

ARCH="${ARCH:-resnet50}"
LOSS="${LOSS:-ce}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.1}"
LR_SCHED="${LR_SCHED:-cosine}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DMIX_RATIO="${DMIX_RATIO:-1.0}"
WD="${WD:-0.0005}"
STAGE2="${STAGE2:-real_ft}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-5}"
STAGE2_SAMPLES_PER_CLASS="${STAGE2_SAMPLES_PER_CLASS:-20}"

GEN_TAG="h2tc_${DATASET}_IF${IMB_FACTOR}_${H2T_VARIANT}"
GEN_TAG="${GEN_TAG}_${FEAT_SRC}_${HEAD_SEL}_${SAMPLE_SEL}_k${TOP_K}"
GEN_TAG="${GEN_TAG}_ht${HEAD_THRESH}_tt${TAIL_THRESH}_s${STRENGTH}_g${GUIDANCE}_st${STEPS}"
GEN_TAG="${GEN_TAG}_${PIPE_TYPE}_${PLAN_MODE}_${PER_IMAGE_SCOPE}_tn${TARGET_NUM}_max${MAX_AUG_PER_CLASS}"
GEN_TAG="${GEN_TAG}_p${N_PROMPTS}_seed${SEED}"
if [ "${ENABLE_LORA}" = "1" ]; then
    GEN_TAG="${GEN_TAG}_${LORA_NAME}"
fi
if [ -n "${RUN_SUFFIX}" ]; then
    GEN_TAG="${GEN_TAG}_${RUN_SUFFIX}"
fi
if [ ${#GEN_TAG} -gt 160 ]; then
    HASH=$(echo "${GEN_TAG}" | md5sum | cut -c1-8)
    GEN_TAG="${GEN_TAG:0:140}_${HASH}"
fi

OUTPUT_DIR="./data/${GEN_TAG}"
BASELINE_TAG="baseline_${DATASET}_IF${IMB_FACTOR}_${ARCH}_${LOSS}_ep${EPOCHS}_lr${LR}_${LR_SCHED}_bs${BATCH_SIZE}_wd${WD}_seed${SEED}"
H2T_TAG="${GEN_TAG}_${ARCH}_${LOSS}_ep${EPOCHS}_lr${LR}_${LR_SCHED}_bs${BATCH_SIZE}_wd${WD}_dr${DMIX_RATIO}_${STAGE2}"
if [ ${#H2T_TAG} -gt 200 ]; then
    HASH2=$(echo "${H2T_TAG}" | md5sum | cut -c1-8)
    H2T_TAG="${H2T_TAG:0:180}_${HASH2}"
fi

echo "Variant: ${H2T_VARIANT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "LoRA weights: ${LORA_WEIGHTS:-none}"
echo "Training exp: ${H2T_TAG}"

if [ "${RUN_BASELINE}" = "1" ]; then
    python main.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --img_root ${IMG_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --arch ${ARCH} \
        --loss ${LOSS} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --lr_schedule ${LR_SCHED} \
        --batch_size ${BATCH_SIZE} \
        --weight_decay ${WD} \
        --augment randaug \
        --warmup_epochs 5 \
        --gpu ${GPU} \
        --output_dir ./output \
        --exp_name ${BASELINE_TAG} \
        --seed ${SEED}
fi

GEN_ARGS=()
if [ -n "${LORA_WEIGHTS}" ]; then
    GEN_ARGS+=(--lora_weights "${LORA_WEIGHTS}")
fi
if [ "${ENABLE_FILTER}" = "1" ]; then
    GEN_ARGS+=(--enable_filter)
fi
if [ "${SOURCE_NEGATIVE_PROMPT}" = "1" ]; then
    GEN_ARGS+=(--source_negative_prompt)
fi

if [ -f "${OUTPUT_DIR}/metadata.json" ]; then
    echo "Found existing controlled Head2Tail metadata at ${OUTPUT_DIR}/metadata.json, skipping generation."
else
    python generate_head2tail_controlled.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --img_root ${IMG_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --output_dir ${OUTPUT_DIR} \
        --top_k ${TOP_K} \
        --head_threshold ${HEAD_THRESH} \
        --tail_threshold ${TAIL_THRESH} \
        --feature_source ${FEAT_SRC} \
        --head_selection ${HEAD_SEL} \
        --sample_selection ${SAMPLE_SEL} \
        --model_id "${MODEL_ID}" \
        --pipeline_type ${PIPE_TYPE} \
        --strength ${STRENGTH} \
        --guidance_scale ${GUIDANCE} \
        --num_inference_steps ${STEPS} \
        --gen_size ${GEN_SIZE} \
        --save_size ${SAVE_SIZE} \
        --n_prompts_per_class ${N_PROMPTS} \
        --plan_mode ${PLAN_MODE} \
        --per_image_scope ${PER_IMAGE_SCOPE} \
        --target_num ${TARGET_NUM} \
        --max_aug_per_class ${MAX_AUG_PER_CLASS} \
        --target_text_threshold ${TARGET_TEXT_THRESHOLD} \
        --prototype_margin_threshold ${PROTOTYPE_MARGIN_THRESHOLD} \
        --keep_top_k_per_class ${KEEP_TOP_K_PER_CLASS} \
        --keep_ratio ${KEEP_RATIO} \
        --max_attempt_multiplier ${MAX_ATTEMPT_MULTIPLIER} \
        --seed ${SEED} \
        "${GEN_ARGS[@]}"
fi

if [ "${RUN_TRAIN}" = "1" ]; then
    python main.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --img_root ${IMG_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --arch ${ARCH} \
        --loss ${LOSS} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --lr_schedule ${LR_SCHED} \
        --batch_size ${BATCH_SIZE} \
        --weight_decay ${WD} \
        --augment randaug \
        --warmup_epochs 5 \
        --gpu ${GPU} \
        --diffusemix_dir ${OUTPUT_DIR} \
        --diffusemix_ratio ${DMIX_RATIO} \
        --use_orig_cls_num \
        --stage2 ${STAGE2} \
        --stage2_epochs ${STAGE2_EPOCHS} \
        --stage2_samples_per_class ${STAGE2_SAMPLES_PER_CLASS} \
        --output_dir ./output \
        --exp_name ${H2T_TAG} \
        --seed ${SEED}
fi

echo "Controlled ImageNet-LT Head2Tail job complete."
