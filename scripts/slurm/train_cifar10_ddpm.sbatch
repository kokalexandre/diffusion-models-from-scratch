#!/bin/bash
#SBATCH --job-name=ddpm-cifar10
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

CONFIG="configs/cifar10_ddpm.yaml"
CKPT_DIR="checkpoints/cifar10_ddpm"

RESUME_ARGS=()
LATEST_CKPT=""

if [ -d "$CKPT_DIR" ]; then
  LATEST_CKPT="$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'ddpm_cifar10_step_*.pt' | sort -V | tail -n 1 || true)"
fi

if [ -n "${LATEST_CKPT:-}" ]; then
  echo "Found checkpoint: $LATEST_CKPT"
  RESUME_ARGS=(--resume "$LATEST_CKPT")
else
  echo "No checkpoint found. Starting from scratch."
fi

micromamba run -n diffusion python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') else 'n/a')"

micromamba run -n diffusion python -u train_cifar10.py \
  --config "$CONFIG" \
  "${RESUME_ARGS[@]}"