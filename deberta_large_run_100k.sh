#!/bin/bash --login

#SBATCH --job-name=dBl_T

#SBATCH --output=logs/all_lms_100k_train_logs/out_deberta_large_100k_mscg_8k_prefix_adj_100k_gkb_prop.txt
#SBATCH --error=logs/all_lms_100k_train_logs/err_deberta_large_100k_mscg_8k_prefix_adj_100k_gkb_prop.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
##SBATCH --qos=gpu7d
#SBATCH --mem=15g
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 run_model.py --config_file configs/100k_dataset_experiments/deberta_large_mscg_and_prefix_adj_and_gkb_prop_config.json

echo 'finished!'