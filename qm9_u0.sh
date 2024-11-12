#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=16:00:00
#SBATCH --job-name=qm9_u0
#SBATCH --output stdout.qm9
#SBATCH --error stderr.qm9
#SBATCH --partition=gpu-long

ulimit -s unlimited
module add miniconda/22.11.1-1
eval "$(conda shell.bash hook)"
conda activate /work/pi_pengbai_umass_edu/pinyang_umass_edu-conda/envs/chemprop

srun --unbuffered python ./train.py

exit 0