#!/bin/bash
#SBATCH --job-name=cifar10-sample-ddpm
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/cifar10_ddpm.yaml}"
CKPT="${CKPT:-}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
SEED="${SEED:-0}"
USE_EMA="${USE_EMA:-1}"

if [ -z "${CKPT}" ]; then
  CKPT="$(find checkpoints/cifar10_ddpm -maxdepth 1 -type f -name 'ddpm_cifar10_step_*.pt' | sort -V | tail -n 1 || true)"
fi

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "No checkpoint found."
  exit 1
fi

STEM="$(basename "${CKPT%.pt}")"
OUT="${OUT:-samples/cifar10_ddpm/${STEM}_ddpm_ema.png}"

EXTRA_ARGS=()
if [ "${USE_EMA}" = "1" ]; then
  EXTRA_ARGS+=(--use_ema)
fi

echo "Using checkpoint: ${CKPT}"
echo "Output: ${OUT}"

micromamba run -n diffusion python -u sample_cifar10.py \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${SEED}" \
  --out "${OUT}" \
  "${EXTRA_ARGS[@]}"