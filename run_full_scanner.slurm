#!/usr/bin/env bash

#SBATCH --job-name=full_tss_scanner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-task=1
#SBATCH --time=80:00:00
#SBATCH --output=full_tss_scanner_%j.out
#SBATCH --error=full_tss_scanner_%j.err
#SBATCH --partition=main
#SBATCH --account=sophont
#SBATCH --qos top
#SBATCH --requeue

set -euo pipefail

# Master node configuration
export MASTER_ADDR=$(hostname -I | awk '{print $1}')
export MASTER_PORT=29501

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# Training config
CONFIG_FILE="./dinov2/configs/train/vitg14_reg4_scanner.yaml"
OUTPUT_DIR="/tmp/dryrun_full_scanner"
RESUME="False" # set string to "True" to resume from last checkpoint in OUTPUT_DIR if it exists

# Set Python path for imports
# Provide script path so train.py can attach the right launcher script to WandB.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DINOV2_RUN_SCRIPT="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Clean output directory only when not resuming
if [[ "${RESUME}" == "True" ]]; then
  echo "Resume enabled; preserving ${OUTPUT_DIR}"
  RESUME_FLAG=""
else
  echo "Resume disabled; cleaning ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
  RESUME_FLAG="--no-resume"
fi
mkdir -p "${OUTPUT_DIR}"

echo "[Master Node] Starting training..."
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=1, NPROC_PER_NODE=8"
echo "CONFIG_FILE=${CONFIG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

uv run torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --node_rank 0 \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  dinov2/train/train.py \
  --config-file "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  ${RESUME_FLAG}
