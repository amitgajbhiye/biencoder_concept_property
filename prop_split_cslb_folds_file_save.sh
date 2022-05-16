#!/bin/bash --login

#SBATCH --job-name=PSFS

#SBATCH --output=logs/nn_analysis/out_cslb_prop_splt_fold_file_save.txt
#SBATCH --error=logs/nn_analysis/err_cslb_prop_splt_fold_file_save.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu,gpu_v100
##SBATCH --gres=gpu:1

#SBATCH -t 0-00:30:00
#SBATCH -p dev

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 fine_tune.py --config_file configs/nn_analysis/cslb_prop_split_file_save_config.json

echo 'finished!'