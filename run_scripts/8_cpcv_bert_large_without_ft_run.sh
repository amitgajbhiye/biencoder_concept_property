#!/bin/bash --login
#SBATCH --job-name=bl_w/o_ft

#SBATCH --output=logs/100k_concept_property_split_logs/out_cpcv_bert_baseline_without_finetune.txt
#SBATCH --error=logs/100k_concept_property_split_logs/err_cpcv_bert_baseline_without_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1

#SBATCH -t 0-5:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Running PCV on BERT LARGE Baseline on Mcrae - configs/fine_tune/cpcv_bert_large_without_fine_tune_mcrae_data_config.json' 
python3 fine_tune.py --config_file configs/fine_tune/cpcv_bert_large_without_fine_tune_mcrae_data_config.json

echo 'finished!'