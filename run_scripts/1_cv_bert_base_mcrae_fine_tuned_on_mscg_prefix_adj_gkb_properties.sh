#!/bin/bash --login
#SBATCH --job-name=mag

#SBATCH --output=logs/100k_cv_logs/out_cv_bert_base_mcrae_fine_tuned_on_100k_mscg_8k_prefix_adj_100k_gkb_properties.txt
#SBATCH --error=logs/100k_cv_logs/err_cv_bert_base_mcrae_fine_tuned_on_100k_mscg_8k_prefix_adj_100k_gkb_properties.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5

#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'cross validation bert base fine tune on McRae train set config_file - configs/fine_tune/cv_mcrae_fine_tune_100k_mscg_8k_prefix_adj_100k_gkb_properties_config.json' 

python3 fine_tune.py --config_file configs/fine_tune/cv_mcrae_fine_tune_100k_mscg_8k_prefix_adj_100k_gkb_properties_config.json

echo 'finished!'