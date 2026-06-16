#!/bin/bash
#SBATCH --job-name=cifar10-sample-ddim-memory
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
CKPT="${CKPT:-checkpoints/cifar10_ddpm/ddpm_cifar10_step_0800000.pt}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
DDIM_STEPS="${DDIM_STEPS:-50}"
DDIM_SCHEDULE="${DDIM_SCHEDULE:-quadratic}"
SEED="${SEED:-0}"
USE_EMA="${USE_EMA:-1}"

# --- Hyperparametres du sampler memory -----------------------------------
MODE="${MODE:-c0}"
ETA="${ETA:-0.2}"
LAMBDA_F2="${LAMBDA_F2:-0.3}"
LAMBDA_H="${LAMBDA_H:-1.0}"
LAMBDA_BETA="${LAMBDA_BETA:-0.4}"
LAMBDA_R="${LAMBDA_R:-0.1}"
EPS_R="${EPS_R:-1e-6}"
LAMBDA_D="${LAMBDA_D:-0.5}"
LAMBDA_E="${LAMBDA_E:-0.5}"
LAMBDA_H2="${LAMBDA_H2:-0.5}"
VERBOSE_PD="${VERBOSE_PD:-0}"
NO_CLIP_DENOISED="${NO_CLIP_DENOISED:-0}"

# --- Selection automatique du dernier checkpoint si CKPT vide ------------
if [ -z "${CKPT}" ]; then
  CKPT="$(find checkpoints/cifar10_ddpm -maxdepth 1 -type f -name 'ddpm_cifar10_step_*.pt' | sort -V | tail -n 1 || true)"
fi

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "No checkpoint found at: ${CKPT}"
  exit 1
fi

STEM="$(basename "${CKPT%.pt}")"
if [ "${MODE}" = "c0" ]; then
  TAG="ddim_memory_c0_steps${DDIM_STEPS}_eta${ETA}_lf2${LAMBDA_F2}_lh${LAMBDA_H}_lb${LAMBDA_BETA}_lr${LAMBDA_R}_lh2${LAMBDA_H2}"
else
  TAG="ddim_memory_bary_steps${DDIM_STEPS}_eta${ETA}_lf2${LAMBDA_F2}_lh${LAMBDA_H}_lb${LAMBDA_BETA}_lr${LAMBDA_R}_ld${LAMBDA_D}_le${LAMBDA_E}"
fi
OUT="${OUT:-samples/cifar10_ddpm/${STEM}_${TAG}.png}"

EXTRA_ARGS=()
if [ "${USE_EMA}" = "1" ]; then EXTRA_ARGS+=(--use_ema); fi
if [ "${VERBOSE_PD}" = "1" ]; then EXTRA_ARGS+=(--verbose_pd); fi
if [ "${NO_CLIP_DENOISED}" = "1" ]; then EXTRA_ARGS+=(--no_clip_denoised); fi

echo "Using checkpoint: ${CKPT}"
echo "Mode: ${MODE}"
echo "Output: ${OUT}"
echo "Memory DDIM params: eta=${ETA} lambda_f2=${LAMBDA_F2} lambda_h=${LAMBDA_H} lambda_beta=${LAMBDA_BETA} lambda_r=${LAMBDA_R}"
echo "  (barycenter) lambda_d=${LAMBDA_D} lambda_e=${LAMBDA_E}    (c0) lambda_h2=${LAMBDA_H2}"

micromamba run -n diffusion python -u sample_cifar10_ddim_memory.py \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --num_samples "${NUM_SAMPLES}" \
  --steps "${DDIM_STEPS}" \
  --schedule "${DDIM_SCHEDULE}" \
  --seed "${SEED}" \
  --out "${OUT}" \
  --mode "${MODE}" \
  --eta "${ETA}" \
  --lambda_f2 "${LAMBDA_F2}" \
  --lambda_h "${LAMBDA_H}" \
  --lambda_beta "${LAMBDA_BETA}" \
  --lambda_r "${LAMBDA_R}" \
  --eps_r "${EPS_R}" \
  --lambda_d "${LAMBDA_D}" \
  --lambda_e "${LAMBDA_E}" \
  --lambda_h2 "${LAMBDA_H2}" \
  "${EXTRA_ARGS[@]}"
