#!/usr/bin/env bash
#SBATCH --job-name=thunder
#SBATCH --output=/data/vbelagali/slurm_logs/%j.out
#SBATCH --error=/data/vbelagali/slurm_logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=40:00:00
#SBATCH --partition=n
#SBATCH --account=sophont
#SBATCH --qos=normal

source ~/.bashrc

# Use the thunder repo's venv (it has the `thunder` package); run from nanopath where the script lives.
source /admin/home/vbelagali/codes/thunder/.venv/bin/activate
cd /admin/home/vbelagali/codes/nanopath

export THUNDER_BASE_DATA_FOLDER="/data/vbelagali/data/thunder-data"

start_time=$(date +%s)
SAVE_NAME="jepafino_jf_i_hed08"
CKPT_PATH="/data/hm/nanopath/jepafino/jf_i_hed08/latest.pt"

CUDA_VISIBLE_DEVICES=0 python thunder_eval.py \
    --save_name "${SAVE_NAME}" \
    --ckpt_path "${CKPT_PATH}" \
    --datasets wilds tcga_crc_msi bracs spider_breast &
CUDA_VISIBLE_DEVICES=1 python thunder_eval.py \
    --save_name "${SAVE_NAME}" \
    --ckpt_path "${CKPT_PATH}" \
    --datasets esca tcga_uniform break_his spider_colorectal &
CUDA_VISIBLE_DEVICES=2 python thunder_eval.py \
    --save_name "${SAVE_NAME}" \
    --ckpt_path "${CKPT_PATH}" \
    --datasets patch_camelyon bach mhist spider_skin &
CUDA_VISIBLE_DEVICES=3 python thunder_eval.py \
    --save_name "${SAVE_NAME}" \
    --ckpt_path "${CKPT_PATH}" \
    --datasets tcga_tils crc ccrcc spider_thorax

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

hours=$(( elapsed / 3600 ))
minutes=$(( (elapsed % 3600) / 60 ))
seconds=$(( elapsed % 60 ))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Eval finished!"
printf "⏱  Time: %02dh %02dm %02ds\n" $hours $minutes $seconds
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
