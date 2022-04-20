#!/bin/bash --login

#SBATCH --job-name=blmag

#SBATCH --output=logs/logs/500k_data_logs/out_bert_large_mscg_plus_prefix_adjective_plus_gkb_500k.txt
#SBATCH --error=logs/logs/500k_data_logs/err_bert_large_mscg_plus_prefix_adjective_plus_gkb_500k.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=18g
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Bert Large - Training On configs/500k_dataset_experiments/bert_large_mscg_plus_prefix_adjective_plus_gkb_500k_config.json' 
python3 run_model.py --config_file configs/500k_dataset_experiments/bert_large_mscg_plus_prefix_adjective_plus_gkb_500k_config.json

echo 'finished!'
