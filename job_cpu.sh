#!/bin/bash
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=05:00:00
#SBATCH --output=job_cpu_log/result_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate causvid

df -h | grep scratch > job_cpu_log/df_debug.txt
cd /home/ysunem/26.2/2.27_CausVid/code

BASE="${SCRATCH:-/scratch/peilab/ysunem}"
INPUT_DIR="$BASE/26.2/2.27_CausVid/data/all_mixkit/all_mixkit_videos"
OUTPUT_DIR="$BASE/26.2/2.27_CausVid/data/all_mixkit/all_mixkit_videos_480"

python -u distillation_data/process_mixkit.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --width 832 --height 480 --fps 16 --resume
echo "DEBUG: Python Script completed"

