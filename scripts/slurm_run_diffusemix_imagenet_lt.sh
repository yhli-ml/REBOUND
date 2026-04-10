#!/bin/bash
#SBATCH --partition=lab-chandar
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere
#SBATCH --mem=80G
#SBATCH --time=2-0:00:00
#SBATCH -o /home/mila/j/jerry.huang/code_yuhangli/REBOUND/scripts/slurm-imagenet_lt_diffusemix-%j.out

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

# ImageNet-LT staging
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

# =========================================================
# ImageNet-LT + DiffuseMix reference baseline: Two-stage pipeline
# =========================================================

GPU=${1:-0}
DATA_ROOT="${IMAGENET_LT_ROOT}"
IMG_ROOT="${IMAGENET_ROOT}"
FRACTAL_DIR="./deviantart"
EPOCHS=100
DATASET="imagenet_lt"
ARCH="resnet50"

# ======= Stage 1: Generate DiffuseMix augmented images =======
DIFFUSEMIX_DIR="./data/diffusemix_${DATASET}"
PROMPTS="sunset,Autumn"
AUG_PER_IMG=1  # Per-image DiffuseMix baseline, aligned with Head2Tail plan

echo "==============================================="
echo " Stage 1: DiffuseMix Reference Baseline"
echo " Dataset: ${DATASET}, Plan: per_image (m=${AUG_PER_IMG})"
echo "==============================================="

if [ ! -f "${DIFFUSEMIX_DIR}/metadata.json" ]; then
    python generate_diffusemix.py \
        --dataset ${DATASET} \
        --data_root ${DATA_ROOT} \
        --img_root ${IMG_ROOT} \
        --fractal_dir ${FRACTAL_DIR} \
        --prompts "${PROMPTS}" \
        --plan_mode per_image \
        --aug_per_image ${AUG_PER_IMG} \
        --output_dir ${DIFFUSEMIX_DIR} \
        --gen_size 256 \
        --save_size 0 \
        --device cuda \
        --seed 42
    echo "Stage 1 complete."
else
    echo "Augmented data already exists at ${DIFFUSEMIX_DIR}, skipping generation."
fi

# ======= Stage 2: Train with DiffuseMix augmented data =======
echo ""
echo "==============================================="
echo " Stage 2: Training with DiffuseMix reference baseline"
echo "==============================================="

# 1. ERM + DiffuseMix
echo "[1/6] Training ERM + DiffuseMix..."
python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
    --arch ${ARCH} --loss ce --epochs ${EPOCHS} --batch_size 256 \
    --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
    --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
    --diffusemix_dir ${DIFFUSEMIX_DIR}

# # 2. CB-CE + DiffuseMix
# echo "[2/6] Training CB-CE + DiffuseMix..."
# python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
#     --arch ${ARCH} --loss cb_ce --epochs ${EPOCHS} --batch_size 256 \
#     --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
#     --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
#     --diffusemix_dir ${DIFFUSEMIX_DIR}

# # 3. LDAM-DRW + DiffuseMix
# echo "[3/6] Training LDAM-DRW + DiffuseMix..."
# python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
#     --arch ${ARCH} --loss ldam --use_norm --drw 60 \
#     --epochs ${EPOCHS} --batch_size 256 \
#     --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
#     --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
#     --diffusemix_dir ${DIFFUSEMIX_DIR}

# # 4. Balanced Softmax + DiffuseMix
# echo "[4/6] Training Balanced Softmax + DiffuseMix..."
# python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
#     --arch ${ARCH} --loss balanced_softmax \
#     --epochs ${EPOCHS} --batch_size 256 \
#     --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
#     --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
#     --diffusemix_dir ${DIFFUSEMIX_DIR}

# # 5. Logit Adjustment + DiffuseMix
# echo "[5/6] Training Logit Adjustment + DiffuseMix..."
# python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
#     --arch ${ARCH} --loss logit_adjust --la_tau 1.0 \
#     --epochs ${EPOCHS} --batch_size 256 \
#     --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
#     --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
#     --diffusemix_dir ${DIFFUSEMIX_DIR}

# # 6. cRT + DiffuseMix
# echo "[6/6] Training cRT + DiffuseMix..."
# python main.py --dataset ${DATASET} --data_root ${DATA_ROOT} --img_root ${IMG_ROOT} \
#     --arch ${ARCH} --loss ce --stage2 crt --stage2_epochs 10 \
#     --epochs ${EPOCHS} --batch_size 256 \
#     --lr 0.1 --lr_schedule cosine --warmup_epochs 5 \
#     --augment randaug --weight_decay 5e-4 --gpu ${GPU} \
#     --diffusemix_dir ${DIFFUSEMIX_DIR}

echo "==============================================="
echo " All ImageNet-LT + DiffuseMix experiments done!"
echo "==============================================="
