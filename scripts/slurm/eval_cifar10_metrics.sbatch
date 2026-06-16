#!/bin/bash
#SBATCH --job-name=cifar10-metrics
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

CONFIG="${CONFIG:-configs/cifar10_ddpm.yaml}"
CKPT="${CKPT:-}"

MODE="${MODE:-single}"

SAMPLER="${SAMPLER:-ddim}"
STEPS="${STEPS:-1000}"
SCHEDULE="${SCHEDULE:-quadratic}"
ETA="${ETA:-0.0}"

TABLE_STEPS="${TABLE_STEPS:-10,20,50,100,1000}"
TABLE_ETAS="${TABLE_ETAS:-0.0,0.2,0.5,1.0}"
INCLUDE_DDPM_NOISY="${INCLUDE_DDPM_NOISY:-1}"

NUM_GEN="${NUM_GEN:-10000}"
GEN_BATCH="${GEN_BATCH:-256}"
IS_SPLITS="${IS_SPLITS:-10}"
GEN_SEED="${GEN_SEED:-0}"

REAL_SPLIT="${REAL_SPLIT:-train}"
REAL_BATCH="${REAL_BATCH:-256}"
REAL_MAX="${REAL_MAX:-50000}"
REAL_SEED="${REAL_SEED:-0}"

USE_EMA="${USE_EMA:-1}"
OUT="${OUT:-}"

if [ -z "${CKPT}" ]; then
  CKPT="$(find checkpoints/cifar10_ddpm -maxdepth 1 -type f -name 'ddpm_cifar10_step_*.pt' | sort -V | tail -n 1 || true)"
fi

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "No checkpoint found."
  exit 1
fi

STEM="$(basename "${CKPT%.pt}")"
SAFE_ETA="${ETA//./p}"

if [ -z "${OUT}" ]; then
  if [ "${MODE}" = "single" ]; then
    OUT="runs/cifar10_metrics/${STEM}_${SAMPLER}_steps${STEPS}_eta${SAFE_ETA}_ng${NUM_GEN}.json"
  else
    OUT="runs/cifar10_metrics/${STEM}_table_ng${NUM_GEN}.json"
  fi
fi

EXTRA_ARGS=()
if [ "${USE_EMA}" = "1" ]; then
  EXTRA_ARGS+=(--use_ema)
fi

if [ "${MODE}" = "table" ] && [ "${INCLUDE_DDPM_NOISY}" = "1" ]; then
  EXTRA_ARGS+=(--include_ddpm_noisy)
fi

echo "Using checkpoint: ${CKPT}"
echo "Mode: ${MODE}"
echo "Output: ${OUT}"

micromamba run -n diffusion python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') else 'n/a')"

if [ "${MODE}" = "single" ]; then
  micromamba run -n diffusion python -u tools/eval_cifar10_metrics.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --mode single \
    --sampler "${SAMPLER}" \
    --steps "${STEPS}" \
    --schedule "${SCHEDULE}" \
    --eta "${ETA}" \
    --num_gen "${NUM_GEN}" \
    --gen_batch "${GEN_BATCH}" \
    --is_splits "${IS_SPLITS}" \
    --real_split "${REAL_SPLIT}" \
    --real_batch "${REAL_BATCH}" \
    --real_max "${REAL_MAX}" \
    --real_seed "${REAL_SEED}" \
    --gen_seed "${GEN_SEED}" \
    --out "${OUT}" \
    "${EXTRA_ARGS[@]}"
else
  micromamba run -n diffusion python -u tools/eval_cifar10_metrics.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT}" \
    --mode table \
    --schedule "${SCHEDULE}" \
    --table_steps "${TABLE_STEPS}" \
    --table_etas "${TABLE_ETAS}" \
    --num_gen "${NUM_GEN}" \
    --gen_batch "${GEN_BATCH}" \
    --is_splits "${IS_SPLITS}" \
    --real_split "${REAL_SPLIT}" \
    --real_batch "${REAL_BATCH}" \
    --real_max "${REAL_MAX}" \
    --real_seed "${REAL_SEED}" \
    --gen_seed "${GEN_SEED}" \
    --out "${OUT}" \
    "${EXTRA_ARGS[@]}"
fi