#!/bin/bash --login
#SBATCH --job-name=finetune
#SBATCH --output=logs/100kDataExp/out_finetune.txt
#SBATCH --error=logs/100kDataExp/err_finetune.txt
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=13g
#SBATCH --gres=gpu:1
#SBATCH -t 0-15:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo '6. Running top_100k_mscg_config ' 
python3 fine_tune.py --config_file configs/fine_tune/fine_tune_mcrae_data_config.json

echo 'finished!'