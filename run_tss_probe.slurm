#!/usr/bin/env bash
# Frozen TSS linear probe. This is a sidecar diagnostic and does not touch the
# ongoing pretraining job.
#
# Set one of:
#   CHECKPOINT_PATH=/path/to/eval/training_XXXXX/teacher_checkpoint.pth
#   CHECKPOINT_DIR=/path/to/OpenMidnight/output_full_adv_lam50
# or:
#   TORCHHUB_MODEL=dinov2_vitg14_reg

#SBATCH --job-name=tss_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --output=tss_probe_%j.out
#SBATCH --error=tss_probe_%j.err
#SBATCH --partition=main
#SBATCH --account=sophont
#SBATCH --qos top

set -euo pipefail

REPO_ROOT="/admin/home/jb.schiratti/OpenMidnight"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_FILE="${CONFIG_FILE:-dinov2/configs/train/vitg14_reg4.yaml}"
SAMPLE_LIST_PATH="${SAMPLE_LIST_PATH:-/block/TCGA/sample_dataset_30.txt}"
OUTPUT_JSON="${OUTPUT_JSON:-output_full_adv_lam50/tss_probe.json}"
MAX_WSIS_PER_CLASS="${MAX_WSIS_PER_CLASS:-10}"
MAX_TILES_PER_WSI="${MAX_TILES_PER_WSI:-2}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-42}"

MODEL_ARGS=()
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  MODEL_ARGS+=(--checkpoint-path "${CHECKPOINT_PATH}")
elif [[ -n "${CHECKPOINT_DIR:-}" ]]; then
  MODEL_ARGS+=(--checkpoint-dir "${CHECKPOINT_DIR}")
elif [[ -n "${TORCHHUB_MODEL:-}" ]]; then
  MODEL_ARGS+=(--torchhub-model "${TORCHHUB_MODEL}")
else
  echo "Set CHECKPOINT_PATH, CHECKPOINT_DIR, or TORCHHUB_MODEL" >&2
  exit 2
fi

echo "Config:       ${CONFIG_FILE}"
echo "Sample list:  ${SAMPLE_LIST_PATH}"
echo "Output JSON:  ${OUTPUT_JSON}"
echo "Model args:   ${MODEL_ARGS[*]}"

uv run python tools/tss_probe.py \
  --config-file "${CONFIG_FILE}" \
  --sample-list-path "${SAMPLE_LIST_PATH}" \
  --output-json "${OUTPUT_JSON}" \
  --max-wsis-per-class "${MAX_WSIS_PER_CLASS}" \
  --max-tiles-per-wsi "${MAX_TILES_PER_WSI}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  "${MODEL_ARGS[@]}"
