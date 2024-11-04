#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=336:00:00
#SBATCH --job-name=latent
#SBATCH --output stdout.latent
#SBATCH --error stderr.latent
#SBATCH --partition=gpu
#SBATCH -q long

ulimit -s unlimited
module add conda/latest
eval "$(conda shell.bash hook)"
conda activate /work/pi_pengbai_umass_edu/pinyang_umass_edu-conda/envs/chemprop

srun --unbuffered chemprop fingerprint --test-path ~/ALMS/data/c1_c20.csv \
    --model-path model_0/checkpoints/best-epoch=10-val_loss=1293782.12.ckpt \
    --ffn-block-index -1 \
    --molecule-featurizers v1_rdkit_2d_normalized \
    --output fps_rdkit_2d.csv
exit 0