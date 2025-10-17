#! /bin/bash
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH -p gpu --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=act_exp
#SBATCH --time=500:00:00
#SBATCH --output=slurm_out/%j.out

cd /home/zyang/IS

python test.py
