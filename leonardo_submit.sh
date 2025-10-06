#!/bin/bash
#SBATCH --job-name=paconet-train
#SBATCH --partition=boost_usr_prod           # Leonardo GPU (Booster) partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8                    # adjust to your dataloader needs
#SBATCH --gres=gpu:1                         # number of A100 GPUs
#SBATCH --time=08:00:00                      # hh:mm:ss
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --qos=normal

# --- safety & prep ---
set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) in $PWD"
mkdir -p logs

module purge

# --- Try to load a Python module automatically ---
load_any_python() {
  # Try any python/<ver> first
  PY_MOD="$(module -t avail 2>&1 | grep -E '^python/' | head -n1 || true)"
  if [ -n "${PY_MOD:-}" ]; then
    module load "$PY_MOD" && return 0
  fi
  # Try common Conda stacks if site provides them
  module load anaconda 2>/dev/null && return 0 || true
  module load miniconda3 2>/dev/null && return 0 || true
  return 1
}

if ! load_any_python; then
  echo "No python module found; will use system python if present."
fi

if ! command -v python >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    ln -s "$(command -v python3)" "$TMPDIR/python"
    PATH="$TMPDIR:$PATH"
  else
    echo "ERROR: No Python found on PATH." >&2
    exit 1
  fi
fi

# --- Create / reuse venv in $WORK ---
VENV_ROOT="${WORK:-$HOME}/.venvs"
VENV_NAME="paconet"
VENV_PATH="$VENV_ROOT/$VENV_NAME"
mkdir -p "$VENV_ROOT"

if [ ! -d "$VENV_PATH" ]; then
  echo "Creating venv at $VENV_PATH"
  python -m venv "$VENV_PATH"
fi
# shellcheck disable=SC1090
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# --- run your project ---
# Your pc.sh ends with the training command:
#   python src/pc/unet_train.py --cfg src/pc/config/train_config.yaml --batch_size 8 --num_epochs 80
bash pc.sh
