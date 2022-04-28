#!/bin/bash --login

#SBATCH --job-name=bbmag

#SBATCH --output=logs/logs/500k_data_logs/concept_property_split_fine_tuned_logs/out_mcrae_ft_bert_base_mscg_plus_prefix_adjective_plus_gkb_500k.txt
#SBATCH --error=logs/logs/500k_data_logs/concept_property_split_fine_tuned_logs/err_mcrae_ft_bert_base_mscg_plus_prefix_adjective_plus_gkb_500k.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=12g
#SBATCH --gres=gpu:1

#SBATCH -t 0-08:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Bert Base - Training on configs/500k_fine_tune/cpcv_bert_base_500k_mscg_plus_prefix_adjective_plus_gkb.json' 
python3 fine_tune.py --config_file configs/500k_fine_tune/cpcv_bert_base_500k_mscg_plus_prefix_adjective_plus_gkb.json

echo 'finished!'