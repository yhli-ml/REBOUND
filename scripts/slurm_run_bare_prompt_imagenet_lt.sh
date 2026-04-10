#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mem=30G
#SBATCH --time=2-0:00:00
#SBATCH -o /home/mila/j/jerry.huang/code_yuhangli/REBOUND/scripts/slurm-imagenet_lt_bareprompt-%j.out

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
IMAGENET_LT_ROOT="${SLURM_TMPDIR}/ImageNet_LT"
IMAGENET_LT_TXT_SOURCE="${IMAGENET_LT_TXT_SOURCE:-${PROJECT_ROOT}/data/ImageNet_LT}"

mkdir -p "${IMAGENET_ROOT}/train"
mkdir -p "${IMAGENET_ROOT}/val"
mkdir -p "${IMAGENET_LT_ROOT}"

cp /network/datasets/imagenet/ILSVRC2012_img_train.tar "$SLURM_TMPDIR/"
tar -xf "$SLURM_TMPDIR/ILSVRC2012_img_train.tar" -C "${IMAGENET_ROOT}/train"
cd "${IMAGENET_ROOT}/train"
find . -name "*.tar" | while read -r NAME ; do
  mkdir -p "${NAME%.tar}"
  tar -xf "${NAME}" -C "${NAME%.tar}"
  rm -f "${NAME}"
done

cp /network/datasets/imagenet/ILSVRC2012_img_val.tar "$SLURM_TMPDIR/"
tar -xf "$SLURM_TMPDIR/ILSVRC2012_img_val.tar" -C "${IMAGENET_ROOT}/val"
cd "${IMAGENET_ROOT}/val"
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

for split in train val test; do
  src_file="${IMAGENET_LT_TXT_SOURCE}/ImageNet_LT_${split}.txt"
  dst_file="${IMAGENET_LT_ROOT}/ImageNet_LT_${split}.txt"
  if [ ! -f "${src_file}" ]; then
    echo "Missing ImageNet-LT annotation file: ${src_file}" >&2
    echo "Set IMAGENET_LT_TXT_SOURCE to the directory containing ImageNet_LT_{train,val,test}.txt" >&2
    exit 1
  fi
  cp "${src_file}" "${dst_file}"
done

cd "${PROJECT_ROOT}"

# ===============================================================
# Bare prompt-controlled diffusion baseline for ImageNet-LT
#   GEN_MODE=txt2img: prompt-only generation
#   GEN_MODE=img2img: same-class real image init + minimal class prompt
#   SCOPE=all or medium_tail
# ===============================================================

DATASET="imagenet_lt"
DATA_ROOT="${IMAGENET_LT_ROOT}"
IMG_ROOT="${IMAGENET_ROOT}"
IMB_FACTOR="${IMB_FACTOR:-0.01}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"

GEN_MODE="${GEN_MODE:-txt2img}"
SCOPE="${SCOPE:-medium_tail}"
AUG_PER_IMG="${AUG_PER_IMG:-1}"
MODEL_ID="runwayml/stable-diffusion-v1-5"
STRENGTH="${STRENGTH:-0.6}"
GUIDANCE="${GUIDANCE:-7.5}"
STEPS="${STEPS:-50}"
GEN_SIZE="${GEN_SIZE:-512}"
SAVE_SIZE="${SAVE_SIZE:-256}"
RUN_BASELINE="${RUN_BASELINE:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

ARCH="resnet50"
LOSS="ce"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-0.1}"
LR_SCHED="${LR_SCHED:-cosine}"
BATCH_SIZE="${BATCH_SIZE:-256}"
WD="${WD:-0.0005}"
DMIX_RATIO="${DMIX_RATIO:-1.0}"

GEN_TAG="bare_${DATASET}_IF${IMB_FACTOR}_${GEN_MODE}_${SCOPE}_m${AUG_PER_IMG}"
if [ "${GEN_MODE}" = "img2img" ]; then
    GEN_TAG="${GEN_TAG}_s${STRENGTH}"
fi
GEN_TAG="${GEN_TAG}_g${GUIDANCE}_st${STEPS}_seed${SEED}"
if [ -n "${RUN_SUFFIX}" ]; then
    GEN_TAG="${GEN_TAG}_${RUN_SUFFIX}"
fi
if [ ${#GEN_TAG} -gt 120 ]; then
    HASH=$(echo "${GEN_TAG}" | md5sum | cut -c1-8)
    GEN_TAG="${GEN_TAG:0:100}_${HASH}"
fi

OUTPUT_DIR="./data/${GEN_TAG}"
BASELINE_TAG="baseline_${DATASET}_IF${IMB_FACTOR}_${ARCH}_${LOSS}_ep${EPOCHS}_lr${LR}_${LR_SCHED}_bs${BATCH_SIZE}_wd${WD}_seed${SEED}"
BARE_TAG="${GEN_TAG}_${ARCH}_${LOSS}_ep${EPOCHS}_lr${LR}_${LR_SCHED}_bs${BATCH_SIZE}_wd${WD}_dr${DMIX_RATIO}"

echo "Output dir: ${OUTPUT_DIR}"
echo "Baseline exp: ${BASELINE_TAG}"
echo "BarePrompt exp: ${BARE_TAG}"
echo "Mode: ${GEN_MODE}, scope: ${SCOPE}, m=${AUG_PER_IMG}"

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

if [ -f "${OUTPUT_DIR}/metadata.json" ]; then
    echo "Found existing BarePrompt metadata at ${OUTPUT_DIR}/metadata.json, skipping generation."
else
    python generate_bare_prompt_diffusion.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --img_root ${IMG_ROOT} \
        --imb_factor ${IMB_FACTOR} \
        --generation_mode ${GEN_MODE} \
        --model_id ${MODEL_ID} \
        --strength ${STRENGTH} \
        --guidance_scale ${GUIDANCE} \
        --num_inference_steps ${STEPS} \
        --gen_size ${GEN_SIZE} \
        --save_size ${SAVE_SIZE} \
        --per_image_scope ${SCOPE} \
        --aug_per_image ${AUG_PER_IMG} \
        --output_dir ${OUTPUT_DIR} \
        --device cuda \
        --seed ${SEED}
fi

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
    --output_dir ./output \
    --exp_name ${BARE_TAG} \
    --seed ${SEED}

echo "BarePrompt ImageNet-LT run complete."
