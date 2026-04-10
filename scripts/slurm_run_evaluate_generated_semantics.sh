#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mem=20G
#SBATCH --time=0-6:00:00
#SBATCH -o /home/mila/j/jerry.huang/code_yuhangli/REBOUND/scripts/slurm-generated_semantics-%j.out

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
# Semantic quality evaluation for generated images
# ===============================================================
#
# Default: compare CIFAR-100-LT Head2Tail, BarePrompt, and DiffuseMix.
#
# Example:
#   sbatch scripts/slurm_run_evaluate_generated_semantics.sh
#
# Custom CIFAR comparison:
#   GEN_DIRS="./data/h2t_dir ./data/bare_dir" \
#   METHOD_NAMES="h2t bare" \
#   OUT_DIR="./eval_generated_semantics/my_compare" \
#   sbatch scripts/slurm_run_evaluate_generated_semantics.sh
#
# ImageNet-LT comparison with staged ImageNet images:
#   DATASET=imagenet_lt \
#   DATA_ROOT=./data/ImageNet_LT \
#   GEN_DIRS="./data/h2t_imagenet_dir ./data/bare_imagenet_dir" \
#   METHOD_NAMES="h2t bare" \
#   OUT_DIR="./eval_generated_semantics/imagenet_lt_compare" \
#   sbatch scripts/slurm_run_evaluate_generated_semantics.sh
#
# If ImageNet is already available somewhere:
#   DATASET=imagenet_lt IMG_ROOT=/path/to/imagenet STAGE_IMAGENET=0 sbatch ...
# ===============================================================

DATASET="${DATASET:-cifar100_lt}"
DATA_ROOT="${DATA_ROOT:-}"
IMG_ROOT="${IMG_ROOT:-}"
IMB_FACTOR="${IMB_FACTOR:-0.01}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
DEVICE="${DEVICE:-cuda}"
CLIP_MODEL="${CLIP_MODEL:-ViT-B/32}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_GEN_PER_METHOD="${MAX_GEN_PER_METHOD:-0}"
MAX_REAL_PER_CLASS="${MAX_REAL_PER_CLASS:-1000}"
MAX_GRID_IMAGES="${MAX_GRID_IMAGES:-120}"
SKIP_REAL_PROTOTYPES="${SKIP_REAL_PROTOTYPES:-0}"
STAGE_IMAGENET="${STAGE_IMAGENET:-1}"

if [ "${DATASET}" = "cifar100_lt" ]; then
    DATA_ROOT="${DATA_ROOT:-./data}"
    DEFAULT_GEN_DIRS="./data/h2t_cifar100_lt_IF0.01_clip_image_nearest_random_k3_ht100_tt20_s0.6_g7.5_st50_img2img_per_image_all_m1_p5_seed42_lora ./data/bare_cifar100_lt_IF0.01_txt2img_all_m1_g7.5_st50_seed42 ./data/diffusemix_cifar100_lt_IF100"
    DEFAULT_METHOD_NAMES="h2t bare diffusemix"
    DEFAULT_OUT_DIR="./eval_generated_semantics/cifar100_compare"
elif [ "${DATASET}" = "imagenet_lt" ]; then
    DATA_ROOT="${DATA_ROOT:-./data/ImageNet_LT}"
    DEFAULT_GEN_DIRS="./data/h2t_imagenet_lt_IF0.01_clip_image_nearest_random_k3_ht100_tt20_s0.6_g7.5_st50_img2img_per_image_medium_tail_m1_p5_seed42 ./data/bare_imagenet_lt_IF0.01_txt2img_medium_tail_m1_g7.5_st50_seed42"
    DEFAULT_METHOD_NAMES="h2t bare"
    DEFAULT_OUT_DIR="./eval_generated_semantics/imagenet_lt_compare"
else
    echo "Unsupported DATASET=${DATASET}. Use cifar100_lt or imagenet_lt."
    exit 1
fi

GEN_DIRS="${GEN_DIRS:-${DEFAULT_GEN_DIRS}}"
METHOD_NAMES="${METHOD_NAMES:-${DEFAULT_METHOD_NAMES}}"
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_DIR}}"

if [ "${DATASET}" = "imagenet_lt" ] && [ "${SKIP_REAL_PROTOTYPES}" != "1" ] && [ -z "${IMG_ROOT}" ]; then
    if [ "${STAGE_IMAGENET}" != "1" ]; then
        echo "IMG_ROOT is required for ImageNet-LT unless SKIP_REAL_PROTOTYPES=1 or STAGE_IMAGENET=1."
        exit 1
    fi

    IMAGENET_ROOT="${SLURM_TMPDIR}/imagenet"
    IMG_ROOT="${IMAGENET_ROOT}"
    mkdir -p "${IMAGENET_ROOT}/train"
    mkdir -p "${IMAGENET_ROOT}/val"

    echo "Staging ImageNet into ${IMAGENET_ROOT}"
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
fi

read -r -a GEN_DIR_ARRAY <<< "${GEN_DIRS}"
read -r -a METHOD_ARRAY <<< "${METHOD_NAMES}"

if [ "${#GEN_DIR_ARRAY[@]}" -ne "${#METHOD_ARRAY[@]}" ]; then
    echo "GEN_DIRS and METHOD_NAMES must have the same number of entries."
    echo "GEN_DIRS=${GEN_DIRS}"
    echo "METHOD_NAMES=${METHOD_NAMES}"
    exit 1
fi

for gen_dir in "${GEN_DIR_ARRAY[@]}"; do
    if [ ! -f "${gen_dir}/metadata.json" ]; then
        echo "Missing metadata.json in ${gen_dir}"
        exit 1
    fi
done

echo "Dataset: ${DATASET}"
echo "Data root: ${DATA_ROOT}"
echo "Image root: ${IMG_ROOT}"
echo "Generated dirs: ${GEN_DIRS}"
echo "Method names: ${METHOD_NAMES}"
echo "Output dir: ${OUT_DIR}"

CMD=(
    python evaluate_generated_semantics.py
    --dataset "${DATASET}"
    --data_root "${DATA_ROOT}"
    --imb_factor "${IMB_FACTOR}"
    --generated_dirs "${GEN_DIR_ARRAY[@]}"
    --method_names "${METHOD_ARRAY[@]}"
    --output_dir "${OUT_DIR}"
    --device "${DEVICE}"
    --clip_model "${CLIP_MODEL}"
    --batch_size "${BATCH_SIZE}"
    --max_gen_per_method "${MAX_GEN_PER_METHOD}"
    --max_real_per_class "${MAX_REAL_PER_CLASS}"
    --max_grid_images "${MAX_GRID_IMAGES}"
    --seed "${SEED}"
)

if [ "${DATASET}" = "imagenet_lt" ]; then
    CMD+=(--img_root "${IMG_ROOT}")
fi

if [ "${SKIP_REAL_PROTOTYPES}" = "1" ]; then
    CMD+=(--skip_real_prototypes)
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "Generated semantic evaluation complete."
