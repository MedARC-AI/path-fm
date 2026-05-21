#!/usr/bin/env bash

#SBATCH --job-name=sanity_panda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=sanity_check_panda_%j.out
#SBATCH --error=sanity_check_panda_%j.err
#SBATCH --partition=main
#SBATCH --account=sophont
#SBATCH --qos top

set -euo pipefail

EVA_PROBE_DIR="/admin/home/jb.schiratti/eva-probe"
CONFIG_FILE="/admin/home/jb.schiratti/OpenMidnight/eval_configs/panda_small.yaml"

export CHECKPOINT_PATH="/data/OpenMidnight_ckpts/openmidnight_checkpoint.pth"
export DATA_ROOT="/block/eva-data/panda/prostate-cancer-grade-assessment"
export MODEL_NAME="openmidnight"
export EMBEDDINGS_ROOT="${EVA_PROBE_DIR}/data/embeddings/openmidnight/panda_sanity_check"
export OUTPUT_ROOT="${EVA_PROBE_DIR}/logs/openmidnight/sanity_check_panda"
export N_RUNS=5
export N_PATCHES=200
export PREDICT_BATCH_SIZE=64
export N_DATA_WORKERS=8

cd "${EVA_PROBE_DIR}"

echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Data root:  ${DATA_ROOT}"
echo "Output:     ${OUTPUT_ROOT}"

uv run eva predict_fit --config "${CONFIG_FILE}"
