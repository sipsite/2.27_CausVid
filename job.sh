#!/bin/bash
#SBATCH --account=peilab
#SBATCH --partition=preempt
#SBATCH --job-name=wan_dmd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=30:00:00
#SBATCH --output=job_log/log_%j.out

export WANDB_API_KEY="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"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate causvid

cd /home/ysunem/26.2/2.27_CausVid/code
export PYTHONPATH=$PYTHONPATH:.
python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 \
  --master_port=29574 \
  causvid/train_distillation.py \
  --config_path configs/wan_causal_dmd.yaml 