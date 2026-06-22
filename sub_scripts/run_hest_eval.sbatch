#!/usr/bin/env bash
#SBATCH --job-name=OMhest
#SBATCH --output=/data/vbelagali/slurm_logs/%j.out
#SBATCH --error=/data/vbelagali/slurm_logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=40:00:00
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=normal

set -euo pipefail

source ~/.bashrc

# ---- What to evaluate ----
RUN_DIR="/data/vbelagali/vits_run1_bs64_ep100"
ITER="0"                                   # final checkpoint (100 epochs x 1000 iters)
CONFIG_FILE="${RUN_DIR}/config.yaml"
WEIGHTS="${RUN_DIR}/eval/training_${ITER}/teacher_checkpoint.pth"
SAVE_NAME="vits_run1_bs64_ep100_iter${ITER}"

# ---- Repos ----
# eval_hest.py lives in OpenMidnight; the `hest` package + its uv env live in HEST.
OPENMIDNIGHT="/admin/home/vbelagali/codes/OpenMidnight"
HEST_DIR="/admin/home/vbelagali/codes/HEST"

# ---- Bench / output locations ----
BENCH_DATA_ROOT="/data/HEST"                                       # already downloaded
RESULTS_DIR="/data/vbelagali/data/hest-bench/results"
EMBED_DATAROOT="/data/vbelagali/data/hest-bench/embed_data"
OUTPUT_DIR="/data/vbelagali/data/hest-bench/setup_logs/${SAVE_NAME}"  # where setup() dumps config.yaml

# dinov2 is imported via PYTHONPATH (NOT installed into HEST's venv, to avoid
# clobbering HEST's transformers/hf-hub pins). xformers isn't in HEST's env, so
# disable it -> MemEffAttention falls back to standard attention for inference.
export PYTHONPATH="${OPENMIDNIGHT}${PYTHONPATH:+:${PYTHONPATH}}"
export XFORMERS_DISABLED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "RUN_DIR=${RUN_DIR}"
echo "WEIGHTS=${WEIGHTS}"
echo "SAVE_NAME=${SAVE_NAME}"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Run from the HEST project dir so `uv run` resolves HEST's environment
# (hest + torch 2.8/cu129). omegaconf is the one dinov2 dep absent from HEST's
# venv that the eval_hest import chain actually needs; pull it in with --with.
cd "${HEST_DIR}"
uv run \
  --with omegaconf \
  python "${OPENMIDNIGHT}/dinov2/eval/eval_hest.py" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --bench_data_root "${BENCH_DATA_ROOT}" \
    --results_dir "${RESULTS_DIR}" \
    --embed_dataroot "${EMBED_DATAROOT}" \
    --save_name "${SAVE_NAME}" \
    --input_size 224 \
    --batch_size 128 \
    --num_workers 8

echo "DONE. Results under ${RESULTS_DIR}/${SAVE_NAME}"
