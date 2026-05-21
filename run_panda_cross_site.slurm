#!/usr/bin/env bash
# Cross-site PANDA evaluation (fit-only, embeddings must already be extracted).
# Prerequisite: run_panda_eval.slurm completed AND generate_manifests.py has been run.
# Runs both cross-site scenarios (10 folds each) sequentially on a single GPU.
# Protocol: 10-fold CV within training site; all slides from other site are test.

#SBATCH --job-name=panda_cross_site
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=panda_cross_site_%j.out
#SBATCH --error=panda_cross_site_%j.err
#SBATCH --partition=main
#SBATCH --account=sophont
#SBATCH --qos top

set -euo pipefail

EVA_PROBE_DIR="/admin/home/jb.schiratti/eva-probe"
CONFIGS_DIR="/admin/home/jb.schiratti/OpenMidnight/eval_configs"

export EMBEDDINGS_ROOT="${EVA_PROBE_DIR}/data/embeddings/openmidnight/panda"
export N_PATCHES=5000
export N_DATA_WORKERS=16

cd "${EVA_PROBE_DIR}"

echo "=== Scenario 1: train Radboud, test Karolinska ==="
for FOLD in $(seq 0 9); do
    echo "--- Fold ${FOLD} ---"
    FOLD="${FOLD}" OUTPUT_ROOT="${EVA_PROBE_DIR}/logs/openmidnight/cross_site/radboud_train/fold_${FOLD}" \
    uv run eva fit --config "${CONFIGS_DIR}/panda_cross_site_radboud_train.yaml"
done

echo "=== Scenario 2: train Karolinska, test Radboud ==="
for FOLD in $(seq 0 9); do
    echo "--- Fold ${FOLD} ---"
    FOLD="${FOLD}" OUTPUT_ROOT="${EVA_PROBE_DIR}/logs/openmidnight/cross_site/karolinska_train/fold_${FOLD}" \
    uv run eva fit --config "${CONFIGS_DIR}/panda_cross_site_karolinska_train.yaml"
done
