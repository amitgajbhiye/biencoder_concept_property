#!/bin/bash --login

#SBATCH --job-name=bbmag

#SBATCH --output=logs/cslb_prop_split_100k_fine_tuned_logs/out_cslb_ft_bert_base_mscg_plus_prefix_adjective_plus_gkb_100k.txt
#SBATCH --error=logs/cslb_prop_split_100k_fine_tuned_logs/err_cslb_ft_bert_base_mscg_plus_prefix_adjective_plus_gkb_100k.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu
#SBATCH --mem=12g
#SBATCH --gres=gpu:1

#SBATCH -t 0-20:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Bert Base - Training on configs/fine_tune/cslb_pcv_fine_tune_100k_mscg_8k_prefix_adj_100k_gkb_properties_config.json' 
python3 fine_tune.py --config_file configs/fine_tune/cslb_pcv_fine_tune_100k_mscg_8k_prefix_adj_100k_gkb_properties_config.json

echo 'finished!'