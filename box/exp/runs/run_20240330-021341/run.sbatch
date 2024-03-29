#!/bin/sh
#SBATCH -J exp_20240330-021341 # job name
#SBATCH -o box/exp/runs/run_20240330-021341/%x.%j.out 
#SBATCH -p A6000 # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:4
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --tasks-per-node=1
#SBATCH  --cpus-per-task=16

date
module purge
echo $CONDA_DEFAULT_ENV
echo $CUDA_VISIBLE_DEVICES


cat box/exp/runs/run_20240330-021341/finetune.sh
sh box/exp/runs/run_20240330-021341/finetune.sh 4 box/exp/runs/run_20240330-021341/checkpoints 20240330-021341 exp



date