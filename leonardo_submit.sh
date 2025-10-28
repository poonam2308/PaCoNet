#!/bin/bash
#SBATCH --job-name=paconet-train
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/paconetlogs/%x-%j.out
#SBATCH --qos=normal

set -euo pipefail
mkdir -p slurm_logs/paconetlogs


# Activate your conda env or python virtual env

eval "$(/leonardo_work/EUHPC_D18_087/miniconda/bin/conda shell.bash hook)"
conda activate myenv

# --- Run your training ---
bash pc.sh
