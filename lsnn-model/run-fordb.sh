#!/bin/bash
#SBATCH --account=deep_rc_mem
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=8G
#SBATCH --time=144:00:00

# https://stackoverflow.com/questions/66805098/how-to-run-jobs-in-paralell-using-one-slurm-batch-script

export OMP_NUM_THREADS=12

python -m tools.train_eval --model="all" --dataset=FORDB --epochs=250 --num_cores=12

