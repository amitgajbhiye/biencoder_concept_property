#!/bin/bash --login
#SBATCH --job-name=6Setting
#SBATCH --output=logs/100kDataExp/out_top_100k_mscg.txt
#SBATCH --error=logs/100kDataExp/err_top_100k_mscg.txt
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=13g
#SBATCH --gres=gpu:1
#SBATCH -t 0-10:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo '6. Running top_100k_mscg_config ' 
python3 run_model.py --config_file configs/100k_dataset_experiments/top_100k_mscg_config.json

echo 'finished!'