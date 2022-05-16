#!/bin/bash --login

#SBATCH --job-name=CPFS

#SBATCH --output=logs/nn_analysis/out_mcrae_con_prop_split_fold_file_save.txt
#SBATCH --error=logs/nn_analysis/err_mcrae_con_prop_split_fold_file_save.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=5g
#SBATCH --gres=gpu:1

#SBATCH -t 0-00:30:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 fine_tune.py --config_file configs/nn_analysis/mcrae_con_prop_split_file_save_config.json

echo 'finished!'