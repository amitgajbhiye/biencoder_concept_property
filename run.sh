#!/bin/bash --login
#SBATCH --job-name=2cntx_exp_
#SBATCH --output=logs/70k_2cntx_exp_out.file
#SBATCH --error=logs/70k_2cntx_exp_err.file
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=13g
#SBATCH --gres=gpu:1
#SBATCH -t 0-4:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Running Context 2 Mean Strategy' 
python3 run_model.py --config_file configs/mean_vector_strategy/context_2_mean.json

echo 'finished!'