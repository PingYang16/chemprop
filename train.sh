#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --job-name=qm9_u0
#SBATCH --output stdout.qm9
#SBATCH --error stderr.qm9
#SBATCH --partition=gpu

ulimit -s unlimited
module add conda/latest
eval "$(conda shell.bash hook)"
conda activate /work/pi_pengbai_umass_edu/pinyang_umass_edu-conda/envs/chemprop

srun --unbuffered chemprop train --data-path ./tests/data/regression/mol/qm9/qm9.csv \
    --task-type regression \
    --output-dir . \
    --smiles-column smiles \
    --target-column u0_atom \

exit 0