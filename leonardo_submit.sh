#!/bin/bash
#SBATCH --job-name=paconet-train
#SBATCH --partition=boost_usr_prod           # Leonardo GPU (Booster) partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8                    # adjust to your dataloader needs
#SBATCH --gres=gpu:1                         # number of A100 GPUs
#SBATCH --time=08:00:00                      # hh:mm:ss
#SBATCH --output=slurm_logs/paconetlogs/%x-%j.out
#SBATCH --qos=normal

# --- safety & prep ---
set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) in $PWD"
mkdir -p slurm_logs/paconetlogs

# --- create or reuse a virtualenv in $WORK (recommended by CINECA) ---
PYTHON=$(command -v python3 || true)
if [ -z "$PYTHON" ]; then
  # try site python modules if present
  if module -t avail 2>&1 | grep -qE '^python/'; then
    PY_MOD=$(module -t avail 2>&1 | grep -E '^python/' | head -n1)
    module load "$PY_MOD" 2>/dev/null || true
    PYTHON=$(command -v python3 || command -v python || true)
  fi
fi
if [ -z "$PYTHON" ]; then
  echo "ERROR: No python3 found. Try: module avail python ; module load python/<ver>" >&2
  exit 1
fi
echo "Using $($PYTHON -V)"
# One venv per project is usually best; reuse across jobs to avoid reinstall time.
VENV_ROOT="$WORK/.venvs"
VENV_NAME="paconet"
VENV_PATH="$VENV_ROOT/$VENV_NAME"

mkdir -p "$VENV_ROOT"
if [ ! -d "$VENV_PATH" ]; then
  echo "Creating venv at $VENV_PATH"
  python -m venv "$VENV_PATH"
  source "$VENV_PATH/bin/activate"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
else
  echo "Reusing existing venv at $VENV_PATH"
  source "$VENV_PATH/bin/activate"
  # Optional: ensure deps are up to date for this run
  pip install -r requirements.txt
fi

# Optional: if you use torch that tracks devices/threads:
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- run your project ---
# Your pc.sh ends with the training command:
#   python src/pc/unet_train.py --cfg src/pc/config/train_config.yaml --batch_size 8 --num_epochs 80
bash pc.sh
