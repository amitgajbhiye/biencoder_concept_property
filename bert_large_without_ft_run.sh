#!/bin/bash --login
#SBATCH --job-name=finetune
#SBATCH --output=logs/100k_property_split_logs/out_property_split_bert_large_without_ft_baseline.txt
#SBATCH --error=logs/100k_property_split_logs/err_property_split_bert_large_without_ft_baseline.txt
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1
#SBATCH -t 0-6:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Running Finetuning on Mcrae - configs/fine_tune/bert_large_without_finetune_mcrae_data_config.json' 
python3 fine_tune.py --config_file configs/fine_tune/bert_large_without_finetune_mcrae_data_config.json

echo 'finished!'