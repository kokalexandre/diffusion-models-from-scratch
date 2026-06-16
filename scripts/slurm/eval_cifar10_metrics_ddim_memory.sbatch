#!/bin/bash
#SBATCH --job-name=cifar10-metrics-ddim-memory
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH --signal=SIGUSR1@300

# Memory-DDIM (sampler du superviseur) porte dans l'infra du repo.
#
# Point cle : lambda_f2=0 ramene EXACTEMENT le sampler a DDIM (sans appel
# modele a t=0), donc c'est le baseline DDIM "fair" mesure dans le meme
# pipeline. On le garde dans le grid comme ancre de comparaison.
#
# Deux modes (MODE=c0 ou MODE=barycenter) :
#   - c0         : c_PJ=0, h2 barycentre analytique (lambda_h2), h1 impose.
#   - barycenter : h2, h1 barycentres (lambda_d, lambda_e) des racines PSD.
# Les lambdas de barycentre doivent rester dans [0.1, 0.9] (bords = surface
# det=0, fragile numeriquement).
#
# Tolerance timeout identique a l'eval augmente :
#   SIGUSR1@300 + time_budget + JSON/CSV/MD a chaque ligne + resume auto.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# --- Generales -----------------------------------------------------------
CONFIG="${CONFIG:-configs/cifar10_ddpm.yaml}"
CKPT="${CKPT:-checkpoints/cifar10_ddpm/ddpm_cifar10_step_0800000.pt}"
USE_EMA="${USE_EMA:-1}"
SCHEDULE="${SCHEDULE:-quadratic}"
MODE="${MODE:-c0}"

# --- Budget d'evaluation -------------------------------------------------
NUM_GEN="${NUM_GEN:-3000}"
GEN_BATCH="${GEN_BATCH:-256}"
IS_SPLITS="${IS_SPLITS:-10}"
GEN_SEED="${GEN_SEED:-0}"

REAL_SPLIT="${REAL_SPLIT:-train}"
REAL_BATCH="${REAL_BATCH:-256}"
REAL_MAX="${REAL_MAX:-50000}"
REAL_SEED="${REAL_SEED:-0}"

# --- Grid ----------------------------------------------------------------
TABLE_STEPS="${TABLE_STEPS:-10,20,50,100}"
TABLE_ETAS="${TABLE_ETAS:-0.0,0.2,0.5,1.0}"
# lambda_f2=0 = baseline DDIM (ancre). Puis couplage croissant.
TABLE_LAMBDA_F2="${TABLE_LAMBDA_F2:-0.0,0.3,0.6,0.9}"
TABLE_LAMBDA_H="${TABLE_LAMBDA_H:-0.5,1.0}"
TABLE_LAMBDA_BETA="${TABLE_LAMBDA_BETA:-0.4}"
TABLE_LAMBDA_R="${TABLE_LAMBDA_R:-0.1}"
# barycenter mode (interieur [0.1,0.9])
TABLE_LAMBDA_D="${TABLE_LAMBDA_D:-0.3,0.5,0.7}"
TABLE_LAMBDA_E="${TABLE_LAMBDA_E:-0.3,0.5,0.7}"
# c0 mode (interieur [0.1,0.9])
TABLE_LAMBDA_H2="${TABLE_LAMBDA_H2:-0.3,0.5,0.7}"

SKIP_INVALID="${SKIP_INVALID:-1}"
SKIP_INVALID_THRESHOLD="${SKIP_INVALID_THRESHOLD:-0}"
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-35400}"
NO_RESUME="${NO_RESUME:-0}"

if [ ! -f "${CKPT}" ]; then
  echo "Checkpoint not found: ${CKPT}"
  exit 1
fi

STEM="$(basename "${CKPT%.pt}")"
OUT="${OUT:-runs/cifar10_metrics_ddim_memory/${STEM}_${MODE}_grid_${NUM_GEN}gen.json}"

EXTRA_ARGS=()
if [ "${USE_EMA}" = "1" ]; then EXTRA_ARGS+=(--use_ema); fi
if [ "${SKIP_INVALID}" = "1" ]; then EXTRA_ARGS+=(--skip_invalid --skip_invalid_threshold "${SKIP_INVALID_THRESHOLD}"); fi
if [ "${NO_RESUME}" = "1" ]; then EXTRA_ARGS+=(--no_resume); fi
EXTRA_ARGS+=(--time_budget_seconds "${TIME_BUDGET_SECONDS}")

echo "Using checkpoint: ${CKPT}"
echo "Mode: ${MODE}"
echo "Output: ${OUT}"
echo "Grid: steps=${TABLE_STEPS} eta=${TABLE_ETAS} lf2=${TABLE_LAMBDA_F2}"
echo "      lh=${TABLE_LAMBDA_H} lb=${TABLE_LAMBDA_BETA} lr=${TABLE_LAMBDA_R}"
echo "      (barycenter) ld=${TABLE_LAMBDA_D} le=${TABLE_LAMBDA_E}"
echo "      (c0)         lh2=${TABLE_LAMBDA_H2}"
echo "      num_gen=${NUM_GEN} real_max=${REAL_MAX}"

micromamba run -n diffusion python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

micromamba run -n diffusion python -u tools/eval_cifar10_metrics_ddim_memory.py \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --schedule "${SCHEDULE}" \
  --mode "${MODE}" \
  --table_steps "${TABLE_STEPS}" \
  --table_etas "${TABLE_ETAS}" \
  --table_lambda_f2 "${TABLE_LAMBDA_F2}" \
  --table_lambda_h "${TABLE_LAMBDA_H}" \
  --table_lambda_beta "${TABLE_LAMBDA_BETA}" \
  --table_lambda_r "${TABLE_LAMBDA_R}" \
  --table_lambda_d "${TABLE_LAMBDA_D}" \
  --table_lambda_e "${TABLE_LAMBDA_E}" \
  --table_lambda_h2 "${TABLE_LAMBDA_H2}" \
  --num_gen "${NUM_GEN}" \
  --gen_batch "${GEN_BATCH}" \
  --is_splits "${IS_SPLITS}" \
  --gen_seed "${GEN_SEED}" \
  --real_split "${REAL_SPLIT}" \
  --real_batch "${REAL_BATCH}" \
  --real_max "${REAL_MAX}" \
  --real_seed "${REAL_SEED}" \
  --out "${OUT}" \
  "${EXTRA_ARGS[@]}"
