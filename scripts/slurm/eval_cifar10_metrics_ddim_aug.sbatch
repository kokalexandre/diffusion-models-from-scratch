#!/bin/bash
#SBATCH --job-name=cifar10-metrics-ddim-aug
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#
# Demande a SLURM d'envoyer SIGUSR1 a tous les processus du step 300s (5 min)
# avant la wall-time. Le script Python attrape ce signal, finit l'evaluation
# en cours, ecrit JSON + CSV + MD, puis sort proprement.
# IMPORTANT : pas de prefixe 'B:' ici, sinon seul le bash batch recevrait le
# signal et pas le process python (qui est lance via micromamba run). En
# omettant B:, SLURM envoie le signal au process group entier.
#SBATCH --signal=SIGUSR1@300

# Grille concue pour identifier l'INFLUENCE de chaque parametre augmente
# sur FID/IS, plutot que pour une recherche d'optimum exhaustive.
#
# Strategie (analyse des equations PJ sections 5-8) :
#   - 3 niveaux (extremes + milieu) sur les 3 axes les plus discriminants :
#       lambda_beta (transport de xhat_0 dans y ; a 0 = ablation pure)
#       rho_g       (force du couplage y -> x via f^2_t)
#       lambda_h    (transport de x_t dans y)
#   - 2 niveaux sur les axes a effet secondaire (lambda_r) ou tests du
#     cas general (lambda_c, lambda_h2). Plages volontairement faibles
#     pour rester quasi-PSD.
#
# Volumetrie attendue :
#   - 4 steps * 4 eta * 3*3*3 * 2*2*2 = 3456 combos theoriques
#   - PSD pre-check retire ~12.5% (eta=0 avec lambda_c>0) => ~3024 valides
#   - num_gen=3000, gen_batch=256 => 12 batches par eval
#   - ~10-15s par eval bf16 => 8-12h selon GPU
#
# Tolerance au timeout :
#   1) SLURM envoie SIGUSR1 a 300s avant la wall (voir #SBATCH --signal).
#      Le script python finit l'eval en cours, ecrit JSON+CSV+MD finaux,
#      puis sort. Aucune ligne perdue.
#   2) En ceinture-bretelles, --time_budget_seconds (par defaut 35400s = 9h50)
#      stoppe pareil meme si SIGUSR1 n'est pas recu (cas exotique).
#   3) JSON+CSV+MD sont reecrits a chaque ligne, donc meme un SIGKILL brutal
#      ne perd que la config en cours.
#   4) Resume automatique : a la relance avec le meme OUT, le script lit le
#      CSV existant et skip les configs deja faites. Pour repartir from
#      scratch, ajouter NO_RESUME=1.
#
# Pour pruner / accelerer : reduire les listes (ex. TABLE_LAMBDA_R="0.05"
# pour passer a 1728 combos, ou TABLE_LAMBDA_C="0.0" pour ne tester que
# le cas simple).

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

export PATH="$HOME/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

# --- Generales -----------------------------------------------------------
CONFIG="${CONFIG:-configs/cifar10_ddpm.yaml}"
CKPT="${CKPT:-checkpoints/cifar10_ddpm/ddpm_cifar10_step_0100000.pt}"
USE_EMA="${USE_EMA:-1}"
SCHEDULE="${SCHEDULE:-quadratic}"

# --- Budget d'evaluation (3K samples comme demande) ----------------------
NUM_GEN="${NUM_GEN:-3000}"
GEN_BATCH="${GEN_BATCH:-256}"
IS_SPLITS="${IS_SPLITS:-10}"
GEN_SEED="${GEN_SEED:-0}"

# Real stats: 50K pour garder le FID comparable au CIFAR-10 train complet
REAL_SPLIT="${REAL_SPLIT:-train}"
REAL_BATCH="${REAL_BATCH:-256}"
REAL_MAX="${REAL_MAX:-50000}"
REAL_SEED="${REAL_SEED:-0}"

# --- Grid sur les hypers augmentes ---------------------------------------
# Outer (16 combos) : memes que pour DDIM standard
TABLE_STEPS="${TABLE_STEPS:-10,20,50,100}"
TABLE_ETAS="${TABLE_ETAS:-0.0,0.2,0.5,1.0}"

# Axes principaux (3 niveaux : extremes + milieu)
TABLE_LAMBDA_BETA="${TABLE_LAMBDA_BETA:-0.0,0.5,1.0}"
TABLE_RHO_G="${TABLE_RHO_G:-0.1,0.5,0.9}"
TABLE_LAMBDA_H="${TABLE_LAMBDA_H:-0.0,0.5,1.0}"

# Axes secondaires (2 niveaux : faible vs eleve dans la plage valide)
TABLE_LAMBDA_R="${TABLE_LAMBDA_R:-0.02,0.1}"
TABLE_LAMBDA_C="${TABLE_LAMBDA_C:-0.0,0.002}"
TABLE_LAMBDA_H2="${TABLE_LAMBDA_H2:-0.0,0.02}"

# Pre-filtre PSD : evite les ~12.5% de combos invalides (eta=0 + lc>0)
SKIP_INVALID="${SKIP_INVALID:-1}"
SKIP_INVALID_THRESHOLD="${SKIP_INVALID_THRESHOLD:-0}"

# Soft time budget : sortie propre apres l'eval en cours si la wallclock
# depasse ce montant. 35400s = 9h50, soit 10 min sous la wall SLURM,
# laissant largement le temps de finir une derniere eval (~15s max).
# Mis a 0 pour desactiver (= on s'appuie uniquement sur SIGUSR1).
TIME_BUDGET_SECONDS="${TIME_BUDGET_SECONDS:-35400}"

# Auto-resume depuis le CSV existant si OUT pointe deja sur un fichier.
# Mettre NO_RESUME=1 pour repartir from scratch (le CSV sera ecrase).
NO_RESUME="${NO_RESUME:-0}"

# --- Selection automatique du dernier checkpoint si non fourni -----------
if [ -z "${CKPT}" ]; then
  CKPT="$(find checkpoints/cifar10_ddpm -maxdepth 1 -type f -name 'ddpm_cifar10_step_*.pt' | sort -V | tail -n 1 || true)"
fi

if [ -z "${CKPT}" ] || [ ! -f "${CKPT}" ]; then
  echo "No checkpoint found."
  exit 1
fi

STEM="$(basename "${CKPT%.pt}")"
OUT="${OUT:-runs/cifar10_metrics_ddim_aug/${STEM}_grid_${NUM_GEN}gen.json}"

EXTRA_ARGS=()
if [ "${USE_EMA}" = "1" ]; then
  EXTRA_ARGS+=(--use_ema)
fi
if [ "${SKIP_INVALID}" = "1" ]; then
  EXTRA_ARGS+=(--skip_invalid --skip_invalid_threshold "${SKIP_INVALID_THRESHOLD}")
fi
if [ "${NO_RESUME}" = "1" ]; then
  EXTRA_ARGS+=(--no_resume)
fi
EXTRA_ARGS+=(--time_budget_seconds "${TIME_BUDGET_SECONDS}")

echo "Using checkpoint: ${CKPT}"
echo "Output: ${OUT}"
echo "Grid axes:"
echo "  steps        = ${TABLE_STEPS}"
echo "  etas         = ${TABLE_ETAS}"
echo "  lambda_beta  = ${TABLE_LAMBDA_BETA}    [primary: x_hat_0 transport]"
echo "  rho_g        = ${TABLE_RHO_G}          [primary: y -> x feedback]"
echo "  lambda_h     = ${TABLE_LAMBDA_H}       [primary: x_t transport]"
echo "  lambda_r     = ${TABLE_LAMBDA_R}       [secondary: y self-noise]"
echo "  lambda_c     = ${TABLE_LAMBDA_C}       [general case: alignment defect]"
echo "  lambda_h2    = ${TABLE_LAMBDA_H2}      [general case: y -> y coupling]"
echo "  num_gen=${NUM_GEN}  real_max=${REAL_MAX}  schedule=${SCHEDULE}"
echo "  skip_invalid=${SKIP_INVALID} (threshold=${SKIP_INVALID_THRESHOLD})"

micromamba run -n diffusion python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'bf16', torch.cuda.is_bf16_supported() if torch.cuda.is_available() and hasattr(torch.cuda, 'is_bf16_supported') else 'n/a')"

micromamba run -n diffusion python -u tools/eval_cifar10_metrics_ddim_aug.py \
  --config "${CONFIG}" \
  --ckpt "${CKPT}" \
  --schedule "${SCHEDULE}" \
  --table_steps "${TABLE_STEPS}" \
  --table_etas "${TABLE_ETAS}" \
  --table_lambda_h "${TABLE_LAMBDA_H}" \
  --table_lambda_beta "${TABLE_LAMBDA_BETA}" \
  --table_lambda_r "${TABLE_LAMBDA_R}" \
  --table_lambda_c "${TABLE_LAMBDA_C}" \
  --table_lambda_h2 "${TABLE_LAMBDA_H2}" \
  --table_rho_g "${TABLE_RHO_G}" \
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
