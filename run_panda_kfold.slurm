#!/usr/bin/env bash
# 10-fold CV PANDA evaluation (fit-only, embeddings must already be extracted).
# Prerequisite: run_panda_eval.slurm completed AND generate_manifests.py has been run.
# After this job, run aggregate_kfold_results.py to get mean ± std kappa.

#SBATCH --job-name=panda_kfold
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=panda_kfold_%j.out
#SBATCH --error=panda_kfold_%j.err
#SBATCH --partition=main
#SBATCH --account=sophont
#SBATCH --qos top

set -euo pipefail

EVA_PROBE_DIR="/admin/home/jb.schiratti/eva-probe"
CONFIG="/admin/home/jb.schiratti/OpenMidnight/eval_configs/panda_kfold.yaml"
LOG_DIR="${EVA_PROBE_DIR}/logs/openmidnight/kfold"

export EMBEDDINGS_ROOT="${EVA_PROBE_DIR}/data/embeddings/openmidnight/panda"
export N_PATCHES=5000
export N_DATA_WORKERS=16

cd "${EVA_PROBE_DIR}"

for FOLD in $(seq 0 9); do
    echo "=== Fold ${FOLD} ==="
    FOLD="${FOLD}" OUTPUT_ROOT="${LOG_DIR}/fold_${FOLD}" \
    uv run eva fit --config "${CONFIG}"
done

echo "All folds done. Run tools/panda/aggregate_kfold_results.py (in eva-probe) --log-dir ${LOG_DIR} to summarize."
