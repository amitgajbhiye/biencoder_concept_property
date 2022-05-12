#!/bin/bash --login

#SBATCH --job-name=mcCSbb

#SBATCH --output=logs/mcrae_bert_base_baseline_log/out_mcrae_con_split_bert_base_baseline.txt
#SBATCH --error=logs/mcrae_bert_base_baseline_log/err_mcrae_con_split_bert_base_baseline.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=8g
#SBATCH --gres=gpu:1

#SBATCH -t 0-06:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 fine_tune.py --config_file configs/fine_tune/mcrae_con_split_bert_base_without_fine_tune_baseline_config.json

echo 'finished!'