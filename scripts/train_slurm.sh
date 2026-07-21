#!/usr/bin/env bash
#SBATCH --job-name=sdbe-gda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# This script is for a Slurm cluster. Adjust partition, GPU, modules, and paths
# to match the local cluster configuration before submitting with sbatch.
mkdir -p logs

# Example only; replace or remove according to your cluster.
# module purge
# module load cuda/12.1
# source .venv/bin/activate

SOURCE_FILE="${SOURCE_FILE:-data/dblpv7.mat}"
TARGET_FILE="${TARGET_FILE:-data/citationv1.mat}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"

srun python train.py \
  --source "${SOURCE_FILE}" \
  --target "${TARGET_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --epochs 400 \
  --warmup-epochs 100 \
  --ramp-epochs 80 \
  --runs 5
