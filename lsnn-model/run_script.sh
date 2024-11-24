#!/bin/bash
#SBATCH --output=python.%j.out
#SBATCH --error=python.%j.err
#SBATCH --account=deep_rc_mem
#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=48:00:00

# https://stackoverflow.com/questions/66805098/how-to-run-jobs-in-paralell-using-one-slurm-batch-script

export OMP_NUM_THREADS=4

srun --exclusive -n 1 -c 1 python -m tools.train_eval_all.py --model="all" --dataset=ECG5000 --epochs=5 --is_all_combs=1 &
srun --exclusive -n 1 -c 1 python -m tools.train_eval_all.py --model="all" --dataset=FORDA --epochs=5 --is_all_combs=1 &
srun --exclusive -n 1 -c 1 python -m tools.train_eval_all.py --model="all" --dataset=FORDB --epochs=5 --is_all_combs=1 &
srun --exclusive -n 1 -c 1 python -m tools.train_eval_all.py --model="all" --dataset=WAFER --epochs=5 --is_all_combs=1 &
srun --exclusive -n 1 -c 1 python -m tools.train_eval_all.py --model="all" --dataset=EQUAKES --epochs=5 --is_all_combs=1 &
