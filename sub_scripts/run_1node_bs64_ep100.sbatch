#!/usr/bin/env bash
#SBATCH --job-name=OMvits_bs64_ep100
#SBATCH --output=/data/vbelagali/slurm_logs/%j.out
#SBATCH --error=/data/vbelagali/slurm_logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=infinite
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=normal

set -euo pipefail

# ---- Training config ----
CONFIG_FILE="./dinov2/configs/train/vits14_reg4_bs64_ep100.yaml"
OUTPUT_DIR="/data/vbelagali/vits_run1_bs64_ep100"
RESUME="True"   # "True" to keep OUTPUT_DIR and resume if checkpoints exist

# ---- Repo env ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
export DINOV2_RUN_SCRIPT="${REPO_ROOT}/$(basename "${BASH_SOURCE[0]}")"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Triton/inductor JIT caches must live on node-local disk. HOME and /data are
# FUSE mounts whose non-atomic rename breaks Triton's compile-cache locking,
# causing "cannot open shared object file" races across the 8 ranks.
export TRITON_CACHE_DIR="/tmp/triton-cache-${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor-cache-${SLURM_JOB_ID}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"

# Clean output directory
if [[ "${RESUME}" == "True" ]]; then
  RESUME_FLAG=""
else
  RESUME_FLAG="--no-resume"
  if [[ "${SLURM_PROCID}" == "0" ]]; then
    rm -rf "${OUTPUT_DIR}"
  fi
fi
mkdir -p "${OUTPUT_DIR}"

echo "CONFIG_FILE=${CONFIG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

uv run torchrun \
  --nnodes 1 \
  --nproc_per_node 8 \
  --node_rank 0 \
  dinov2/train/train.py \
  --config-file "${CONFIG_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  ${RESUME_FLAG}
