#!/bin/bash
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --job-name=wan_dmd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --output=job_log/log_inf1_%j.out


source ~/miniconda3/etc/profile.d/conda.sh
conda activate causvid

cd /home/ysunem/ys_26.2/2.27_CausVid/code/misc_code/
export PYTHONPATH=$PYTHONPATH:.
python 3.5/3.5_count_para_and_flop.py

