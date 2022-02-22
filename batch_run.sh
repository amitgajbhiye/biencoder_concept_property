#!/bin/bash --login
#SBATCH --job-name=100kDataExp
#SBATCH --output=logs/100kDataExp/batch_100k_dataset_experiments_out.file
#SBATCH --error=logs/100kDataExp/batch_100k_dataset_experiments_out.file
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1
#SBATCH -t 0-2:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running Experiments...'

# echo '1. Running top_100k_mscg_config Experiment' 
# python3 run_model.py --config_file configs/100k_dataset_experiments/top_100k_mscg_config.json

echo

echo '2. Running top_100k_mscg_and_gkb_config ' 
python3 run_model.py --config_file configs/100k_dataset_experiments/top_100k_mscg_and_gkb_config.json

echo

echo '3. Running top_100k_gkb_ext_prop_config' 
python3 run_model.py --config_file configs/100k_dataset_experiments/top_100k_gkb_ext_prop_config.json

echo

echo '4. Running prefix_augmented_top_100k_mscg_config' 
python3 run_model.py --config_file configs/100k_dataset_experiments/prefix_augmented_top_100k_mscg_config.json

echo

echo '5. Running prefix_adjectives_8k_mscg' 
python3 run_model.py --config_file configs/100k_dataset_experiments/prefix_adjectives_8k_mscg.json


echo '6. Running 100k_mscg_8k_prefix_adj_100k_gkb_properties_config' 
python3 run_model.py --config_file configs/100k_dataset_experiments/100k_mscg_8k_prefix_adj_100k_gkb_properties_config.json

echo 'Finished All Experiments!'
