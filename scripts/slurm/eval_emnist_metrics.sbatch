#!/bin/bash
#SBATCH --job-name=emnist-metrics
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

DDPM_CKPT="checkpoints/emnist_ddpm/ddpm_emnist_epoch_0029.pt"
CLS_CKPT="checkpoints/emnist_classifier/emnist_resnet34_best_balanced.pt"

micromamba run -n diffusion python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

micromamba run -n diffusion python -u tools/eval_emnist_metrics.py \
  --ddpm_config configs/emnist_ddpm.yaml \
  --ddpm_ckpt "$DDPM_CKPT" \
  --use_ema \
  --cls_ckpt "$CLS_CKPT" \
  --num_gen 5000 \
  --gen_batch 64 \
  --real_splits train,test \
  --real_max 50000 \
  --real_seed 0 \
  --out runs/emnist_metrics/metrics_epoch_0029.json