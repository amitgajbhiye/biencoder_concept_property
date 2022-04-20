#!/bin/bash --login

#SBATCH --job-name=a

#SBATCH --output=logs/500k_data_logs/out_mscg_prefix_adj_41k_bert_base.txt
#SBATCH --error=logs/500k_data_logs/err_mscg_prefix_adj_41k_bert_base.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Bert Base - Training on configs/500k_dataset_experiments/mscg_prefix_adj_41k_config.json' 
python3 run_model.py --config_file configs/500k_dataset_experiments/mscg_prefix_adj_41k_config.json

echo 'finished!'