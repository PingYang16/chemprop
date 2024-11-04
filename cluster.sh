#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=336:00:00
#SBATCH --job-name=cluster
#SBATCH --output stdout.cluster
#SBATCH --error stderr.cluster
#SBATCH --partition=gpu
#SBATCH -q long

ulimit -s unlimited
module add conda/latest
eval "$(conda shell.bash hook)"
conda activate /work/pi_pengbai_umass_edu/pinyang_umass_edu-conda/envs/chemprop

srun --unbuffered python latent_representation.py

exit 0