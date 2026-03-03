#!/bin/bash
#SBATCH --job-name=emnist-cls
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

micromamba run -n diffusion python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

micromamba run -n diffusion python -u tools/train_emnist_classifier.py --config configs/emnist_classifier.yaml