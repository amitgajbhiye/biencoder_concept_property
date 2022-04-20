#!/bin/bash --login

#SBATCH --job-name=m

#SBATCH --output=logs/logs/500k_data_logs/default_concept_split_fine_tuned_logs/out_mcrae_ft_mscg_500k_bert_base.txt
#SBATCH --error=logs/logs/500k_data_logs/default_concept_split_fine_tuned_logs/err_mcrae_ft_mscg_500k_bert_base.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-8:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Bert Base - Training on configs/500k_fine_tune/ft_top_500k_mscg_config.json' 
python3 fine_tune.py --config_file configs/500k_fine_tune/ft_top_500k_mscg_config.json

echo 'finished!'