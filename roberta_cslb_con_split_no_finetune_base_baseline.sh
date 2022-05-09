#!/bin/bash --login

#SBATCH --job-name=cs_RB

#SBATCH --output=logs/cslb_fine_tuned_100k_logs/out_con_split_roberta_base_cslb_data_without_fine_tune_baseline.txt
#SBATCH --error=logs/cslb_fine_tuned_100k_logs/err_con_split_roberta_base_cslb_data_without_fine_tune_baseline.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-15:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'ROBERTa Base - No Fine tune Baseline Training On configs/fine_tune/cslb_con_split_roberta_base_without_fine_tune_config.json' 
python3 fine_tune.py --config_file configs/fine_tune/cslb_con_split_roberta_base_without_fine_tune_config.json

echo 'finished!'