#!/bin/bash --login
#SBATCH --job-name=BLarge
#SBATCH --output=logs/100kDataExp/out_bert_large_mscg_and_prefix_adj_and_gkb_properties.txt
#SBATCH --error=logs/100kDataExp/err_bert_large_mscg_and_prefix_adj_and_gkb_properties.txt
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

echo 'bert large Running bert_large_mscg_and_prefix_adj_and_gkb_properties_config' 
python3 run_model.py --config_file configs/100k_dataset_experiments/bert_large_mscg_and_prefix_adj_and_gkb_properties_config.json

echo 'finished!'