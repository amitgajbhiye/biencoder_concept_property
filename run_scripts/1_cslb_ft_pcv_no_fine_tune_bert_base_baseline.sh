#!/bin/bash --login

#SBATCH --job-name=ps_bbmag

#SBATCH --output=logs/cslb_fine_tuned_100k_logs/out_cslb_prop_split_bert_base_baseline.txt
#SBATCH --error=logs/cslb_fine_tuned_100k_logs/err_cslb_prop_split_bert_base_baseline.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Bert prop_cv base - Training On configs/fine_tune/cslb_pcv_bert_base_without_fine_tune_cslb_data_config.json' 
python3 fine_tune.py --config_file configs/fine_tune/cslb_pcv_bert_base_without_fine_tune_cslb_data_config.json

echo 'finished!'