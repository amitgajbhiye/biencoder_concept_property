#!/bin/bash --login
#SBATCH --job-name=mag

#SBATCH --output=logs/100k_property_split_logs/out_ft_bert_base_mcrae_fine_tuned_on_100k_mscg_8k_prefix_adj_100k_gkb_properties.txt
#SBATCH --error=logs/100k_property_split_logs/err_ft_bert_base_mcrae_fine_tuned_on_100k_mscg_8k_prefix_adj_100k_gkb_properties.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5

#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1
#SBATCH -t 0-05:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Fine tuning bert base fine tune on McRae train set config_file - configs/fine_tune/ft_mcrae_fine_tune_100k_mscg_8k_prefix_adj_100k_gkb_properties_config.json' 

python3 fine_tune.py --config_file configs/fine_tune/ft_mcrae_fine_tune_100k_mscg_8k_prefix_adj_100k_gkb_properties_config.json

echo 'finished!'