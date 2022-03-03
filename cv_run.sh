#!/bin/bash --login
#SBATCH --job-name=cvTest
#SBATCH --output=logs/100k_cv_models/out_cv_test.txt
#SBATCH --error=logs/100k_cv_models/err_cv_test.txt
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=13g
#SBATCH --gres=gpu:1
#SBATCH -t 0-01:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'CV Test' 

python3 fine_tune.py --config_file configs/fine_tune/mcrae_fine_tune_100k_mscg_8k_prefix_adj_100k_gkb_properties_config.json

echo 'finished!'