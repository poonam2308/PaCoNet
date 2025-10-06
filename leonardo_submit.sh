#!/bin/bash
#SBATCH --job-name=paconet-train
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/paconetlogs/%x-%j.out
#SBATCH --qos=normal

set -euo pipefail
echo "Job $SLURM_JOB_ID on $(hostname) in $PWD"
mkdir -p slurm_logs/paconetlogs

# --- Python 3.11 from modules (site-provided) ---
module load python/3.11.7
echo "Using $(python3 -V)"

# --- Paths ---
VENV_ROOT="$WORK/.venvs"
VENV_NAME="paconet"
VENV_PATH="$VENV_ROOT/$VENV_NAME"

REQS="$SLURM_SUBMIT_DIR/requirements.txt"   # requirements that live in your project folder
HOME_WHEELHOUSE="$HOME/wheelhouse"          # prebuilt wheels (built on login node)
WORK_WHEELHOUSE="$WORK/wheelhouse"          # optional copy if HOME isn't visible on compute nodes

# Prefer HOME wheelhouse; if not present/mounted, fall back to WORK copy
if [ -d "$HOME_WHEELHOUSE" ]; then
  WHEELHOUSE="$HOME_WHEELHOUSE"
elif [ -d "$WORK_WHEELHOUSE" ]; then
  WHEELHOUSE="$WORK_WHEELHOUSE"
else
  echo "ERROR: No wheelhouse found. Expected $HOME_WHEELHOUSE or $WORK_WHEELHOUSE" >&2
  echo "Build it on the login node with: pip wheel -r $REQS -w ~/wheelhouse" >&2
  exit 2
fi
echo "Using wheelhouse at: $WHEELHOUSE"

# --- Create / reuse venv in $WORK (no internet on compute nodes) ---
mkdir -p "$VENV_ROOT"
if [ ! -d "$VENV_PATH" ]; then
  echo "Creating venv at $VENV_PATH"
  python3 -m venv --copies "$VENV_PATH"
fi

# Activate and ensure up-to-date pip (from wheelhouse if present)
source "$VENV_PATH/bin/activate"

# Try to upgrade pip from local wheels first; if not available, keep existing pip
if ls "$WHEELHOUSE"/pip-*.whl >/dev/null 2>&1 ; then
  python -m pip install --no-index --find-links "$WHEELHOUSE" --upgrade pip
fi

# Install all deps strictly offline from wheelhouse
python -m pip install --no-index --find-links "$WHEELHOUSE" -r "$REQS"

# If you keep extra local wheels (e.g., torch/cu* wheels) in another folder, you can add:
# EXTRA_WHEELS="$HOME/extra_wheels"
# [ -d "$EXTRA_WHEELS" ] && python -m pip install --no-index --find-links "$EXTRA_WHEELS" <extra_pkgs>

# Performance hints
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# --- Run your training ---
bash pc.sh
