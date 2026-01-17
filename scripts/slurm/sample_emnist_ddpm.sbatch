#!/bin/bash
#SBATCH --job-name=ddpm-sample
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"

CKPT="checkpoints/emnist_ddpm/ddpm_emnist_epoch_0049.pt"
OUT="samples/emnist_ddpm/sample_epoch_0049_ema.png"
NUM_SAMPLES=64

micromamba run -n diffusion python -u sample.py \
  --config configs/emnist_ddpm.yaml \
  --ckpt "$CKPT" \
  --num_samples "$NUM_SAMPLES" \
  --use_ema \
  --out "$OUT"
