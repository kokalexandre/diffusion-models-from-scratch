#!/bin/bash
#SBATCH --job-name=cifar10-sample-ddim-aug
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

# --- Generales -----------------------------------------------------------
CONFIG="${CONFIG:-configs/cifar10_ddpm.yaml}"
CKPT="${CKPT:-}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
DDIM_STEPS="${DDIM_STEPS:-50}"
DDIM_SCHEDULE="${DDIM_SCHEDULE:-quadratic}"
SEED="${SEED:-0}"
USE_EMA="${USE_EMA:-1}"

# --- Hyperparametres du sampler augmente (PJ section 8) ------------------
ETA="${ETA:-0.2}"
LAMBDA_H="${LAMBDA_H:-0.5}"
LAMBDA_BETA="${LAMBDA_BETA:-0.5}"
LAMBDA_R="${LAMBDA_R:-0.05}"
EPS_R="${EPS_R:-1e-3}"
LAMBDA_C="${LAMBDA_C:-0.0}"
LAMBDA_H2="${LAMBDA_H2:-0.0}"
RHO_G="${RHO_G:-0.5}"
VERBOSE_PD="${VERBOSE_PD:-0}"
NO_CLIP_DENOISED="${NO_CLIP_DENOISED:-0}"

# --- Selection automatique du dernier checkpoint si non fourni -----------
if [ -z "${CKPT}" ]; then
  CKPT="$(find checkpoints/cifar10_ddpm -maxdepth 1 -type f -name 'ddpm_cifar10_step_*.pt' | sort -V | tail -n 1 || true)"
fi

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "No checkpoint found."
  exit 1
fi

STEM="$(basename "${CKPT%.pt}")"
TAG="ddim_aug_steps${DDIM_STEPS}_eta${ETA}_lh${LAMBDA_H}_lb${LAMBDA_BETA}_lr${LAMBDA_R}_lc${LAMBDA_C}_lh2${LAMBDA_H2}_rho${RHO_G}"
OUT="${OUT:-samples/cifar10_ddpm/${STEM}_${TAG}.png}"

EXTRA_ARGS=()
if [ "${USE_EMA}" = "1" ]; then
  EXTRA_ARGS+=(--use_ema)
fi
if [ "${VERBOSE_PD}" = "1" ]; then
  EXTRA_ARGS+=(--verbose_pd)
fi
if [ "${NO_CLIP_DENOISED}" = "1" ]; then
  EXTRA_ARGS+=(--no_clip_denoised)
fi

echo "Using checkpoint: ${CKPT}"
echo "Output: ${OUT}"
echo "Aug DDIM params: eta=${ETA} lambda_h=${LAMBDA_H} lambda_beta=${LAMBDA_BETA} lambda_r=${LAMBDA_R} lambda_c=${LAMBDA_C} lambda_h2=${LAMBDA_H2} rho_g=${RHO_G}"

micromamba run -n diffusion python -u sample_cifar10_ddim_augmented.py \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --num_samples "${NUM_SAMPLES}" \
  --steps "${DDIM_STEPS}" \
  --schedule "${DDIM_SCHEDULE}" \
  --seed "${SEED}" \
  --out "${OUT}" \
  --eta "${ETA}" \
  --lambda_h "${LAMBDA_H}" \
  --lambda_beta "${LAMBDA_BETA}" \
  --lambda_r "${LAMBDA_R}" \
  --eps_r "${EPS_R}" \
  --lambda_c "${LAMBDA_C}" \
  --lambda_h2 "${LAMBDA_H2}" \
  --rho_g "${RHO_G}" \
  "${EXTRA_ARGS[@]}"
