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

test -f "$DDPM_CKPT"
test -f "$CLS_CKPT"

SAMPLER="ddim"
ETA="0.0"
DDIM_STEPS="1000"
DDIM_SCHEDULE="linear"

NUM_GEN="5000"
GEN_BATCH="512"
REAL_MAX="50000"
REAL_SEED="0"
GEN_SEED="0"

OUT="runs/emnist_metrics/${SAMPLER}_T${DDIM_STEPS}_eta${ETA}_epoch_0029_numgen${NUM_GEN}_realmax${REAL_MAX}_realseed${REAL_SEED}_genseed${GEN_SEED}.json"

micromamba run -n diffusion python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

micromamba run -n diffusion python -u tools/eval_emnist_metrics.py \
  --ddpm_config configs/emnist_ddpm.yaml \
  --ddpm_ckpt "$DDPM_CKPT" \
  --use_ema \
  --cls_ckpt "$CLS_CKPT" \
  --sampler "$SAMPLER" \
  --eta "$ETA" \
  --ddim_steps "$DDIM_STEPS" \
  --ddim_schedule "$DDIM_SCHEDULE" \
  --num_gen "$NUM_GEN" \
  --gen_batch "$GEN_BATCH" \
  --real_splits train,test \
  --real_max "$REAL_MAX" \
  --real_seed "$REAL_SEED" \
  --gen_seed "$GEN_SEED" \
  --out "$OUT"