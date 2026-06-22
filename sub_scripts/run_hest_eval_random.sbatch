#!/usr/bin/env bash
#SBATCH --job-name=OMhestR
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
# Random-init baseline: same arch/config as the run, NO trained weights loaded.
# CONFIG_FILE only fixes the architecture; the model is randomly initialized.
RUN_DIR="/data/vbelagali/vits_run1_bs64_ep100"
CONFIG_FILE="${RUN_DIR}/config.yaml"
SEED="0"
SAVE_NAME="vits_random_init_seed${SEED}"

# ---- Repos ----
# eval_hest_random.py lives in OpenMidnight; the `hest` package + uv env live in HEST.
OPENMIDNIGHT="/admin/home/vbelagali/codes/OpenMidnight"
HEST_DIR="/admin/home/vbelagali/codes/HEST"

# ---- Bench / output locations ----
BENCH_DATA_ROOT="/data/HEST"
RESULTS_DIR="/data/vbelagali/data/hest-bench/results"
EMBED_DATAROOT="/data/vbelagali/data/hest-bench/embed_data"
OUTPUT_DIR="/data/vbelagali/data/hest-bench/setup_logs/${SAVE_NAME}"
mkdir -p "${OUTPUT_DIR}" /data/vbelagali/slurm_logs

# dinov2 is imported via PYTHONPATH (not installed into HEST's venv). xformers
# isn't in HEST's env, so disable it -> standard attention for inference.
export PYTHONPATH="${OPENMIDNIGHT}${PYTHONPATH:+:${PYTHONPATH}}"
export XFORMERS_DISABLED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "CONFIG_FILE=${CONFIG_FILE}  (arch only, random weights)"
echo "SAVE_NAME=${SAVE_NAME}"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cd "${HEST_DIR}"
uv run \
  --with omegaconf \
  python "${OPENMIDNIGHT}/dinov2/eval/eval_hest_random.py" \
    --config-file "${CONFIG_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --seed "${SEED}" \
    --bench_data_root "${BENCH_DATA_ROOT}" \
    --results_dir "${RESULTS_DIR}" \
    --embed_dataroot "${EMBED_DATAROOT}" \
    --save_name "${SAVE_NAME}" \
    --input_size 224 \
    --batch_size 256 \
    --num_workers 8

echo "DONE. Results under ${RESULTS_DIR}/${SAVE_NAME}"
