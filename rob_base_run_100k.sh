#!/bin/bash --login

#SBATCH --job-name=robB_T

#SBATCH --output=logs/roberta_base_100k_model_training_logs/out_rob_base_100k_mscg_8k_prefix_adj_100k_gkb_prop.txt
#SBATCH --error=logs/roberta_base_100k_model_training_logs/err_rob_base_100k_mscg_8k_prefix_adj_100k_gkb_prop.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
##SBATCH --qos=gpu7d
#SBATCH --mem=13g
#SBATCH --gres=gpu:1

#SBATCH -t 1-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Training Roberta Base on 100k_mscg_plus_8k_Prefix_adj_plus_100k_gkb_prop' 
echo 'Training with COnfig file - configs/100k_dataset_experiments/rob_base_100k_mscg_8k_prefix_adj_100k_gkb_properties_config.json'
python3 run_model.py --config_file configs/100k_dataset_experiments/rob_base_100k_mscg_8k_prefix_adj_100k_gkb_properties_config.json

echo 'finished!'