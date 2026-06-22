#!/usr/bin/env bash
#SBATCH --job-name=OMthunder
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
ITER="100000"                                   # final checkpoint (100 epochs x 1000 iters)
CONFIG_FILE="${RUN_DIR}/config.yaml"
WEIGHTS="${RUN_DIR}/eval/training_${ITER}/teacher_checkpoint.pth"
SAVE_NAME="vits_run1_bs64_ep100_iter${ITER}"
TASK="knn"

# ---- Repos ----
# eval_thunder.py lives in OpenMidnight; the `thunder` package + its venv live in THUNDER.
OPENMIDNIGHT="/admin/home/vbelagali/codes/OpenMidnight"
THUNDER_DIR="/admin/home/vbelagali/codes/thunder"

# ---- Bench / output locations ----
export THUNDER_BASE_DATA_FOLDER="/data/vbelagali/data/thunder-data"
OUTPUT_DIR="/data/vbelagali/data/thunder-bench/setup_logs/${SAVE_NAME}"  # where setup() dumps config.yaml

# dinov2 is imported via PYTHONPATH (NOT installed into THUNDER's venv, to avoid
# clobbering its pinned deps). xformers isn't in THUNDER's env, so disable it ->
# MemEffAttention falls back to standard attention for inference.
source "${THUNDER_DIR}/.venv/bin/activate"
export PYTHONPATH="${OPENMIDNIGHT}${PYTHONPATH:+:${PYTHONPATH}}"
export XFORMERS_DISABLED=1
export HDF5_USE_FILE_LOCKING=FALSE

echo "RUN_DIR=${RUN_DIR}"
echo "WEIGHTS=${WEIGHTS}"
echo "SAVE_NAME=${SAVE_NAME}"
echo "node=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

EVAL="${OPENMIDNIGHT}/dinov2/eval/eval_thunder.py"

start_time=$(date +%s)

# Split the 16 datasets across the 4 GPUs on the node, one process per GPU.
# Unset SLURM_JOB_ID per process so dinov2's distributed init takes the local
# path (_set_from_local) and picks a free, distinct master port for each --
# otherwise all 4 derive the same Slurm-seeded port and collide (EADDRINUSE).
env -u SLURM_JOB_ID CUDA_VISIBLE_DEVICES=0 python "${EVAL}" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_name "${SAVE_NAME}" \
    --task "${TASK}" \
    --datasets wilds tcga_crc_msi bracs spider_breast &
env -u SLURM_JOB_ID CUDA_VISIBLE_DEVICES=1 python "${EVAL}" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_name "${SAVE_NAME}" \
    --task "${TASK}" \
    --datasets esca tcga_uniform break_his spider_colorectal &
env -u SLURM_JOB_ID CUDA_VISIBLE_DEVICES=2 python "${EVAL}" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_name "${SAVE_NAME}" \
    --task "${TASK}" \
    --datasets patch_camelyon bach mhist spider_skin &
env -u SLURM_JOB_ID CUDA_VISIBLE_DEVICES=3 python "${EVAL}" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_name "${SAVE_NAME}" \
    --task "${TASK}" \
    --datasets tcga_tils crc ccrcc spider_thorax &
wait

# segmentation and linear probing on subset of datasets

env -u SLURM_JOB_ID CUDA_VISIBLE_DEVICES=0 python "${EVAL}" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_name "${SAVE_NAME}" \
    --task segmentation \
    --datasets pannuke &
env -u SLURM_JOB_ID CUDA_VISIBLE_DEVICES=1 python "${EVAL}" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_name "${SAVE_NAME}" \
    --task segmentation \
    --datasets ocelot &
env -u SLURM_JOB_ID CUDA_VISIBLE_DEVICES=2 python "${EVAL}" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_name "${SAVE_NAME}" \
    --task linear_probing \
    --datasets spider_breast &
env -u SLURM_JOB_ID CUDA_VISIBLE_DEVICES=3 python "${EVAL}" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_name "${SAVE_NAME}" \
    --task linear_probing \
    --datasets spider_colorectal &
env -u SLURM_JOB_ID CUDA_VISIBLE_DEVICES=4 python "${EVAL}" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_name "${SAVE_NAME}" \
    --task linear_probing \
    --datasets spider_skin &
env -u SLURM_JOB_ID CUDA_VISIBLE_DEVICES=5 python "${EVAL}" \
    --config-file "${CONFIG_FILE}" \
    --pretrained-weights "${WEIGHTS}" \
    --output-dir "${OUTPUT_DIR}" \
    --save_name "${SAVE_NAME}" \
    --task linear_probing \
    --datasets spider_thorax
wait

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Eval finished!"
printf "⏱  Time: %02dh %02dm %02ds\n" $hours $minutes $seconds
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "SAVE_NAME=${SAVE_NAME}"
