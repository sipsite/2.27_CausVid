#!/bin/bash
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --job-name=wan_dmd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --time=02:00:00
#SBATCH --output=job_log/log_inf2_%j.out


source ~/miniconda3/etc/profile.d/conda.sh
conda activate causvid

cd /home/ysunem/26.2/2.27_CausVid/code
export PYTHONPATH=$PYTHONPATH:.
python minimal_inference/autoregressive_inference.py \
  --config_path configs/wan_causal_dmd.yaml \
  --checkpoint_folder /scratch/peilab/ysunem/26.2/2.27_CausVid/data/ckpt_ode_pretraining/wan_causal_ode_checkpoint_model_003000 \
  --output_folder jobs_sh/inference_output_ode \
  --prompt_file_path /home/ysunem/26.2/2.27_CausVid/code/sample_dataset/MovieGenVideoBench_8_ys.txt

