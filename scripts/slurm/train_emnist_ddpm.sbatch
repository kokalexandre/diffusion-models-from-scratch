#!/bin/bash
#SBATCH --job-name=ddpm-emnist
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

micromamba run -n diffusion python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

micromamba run -n diffusion python -u train.py --config configs/emnist_ddpm.yaml
