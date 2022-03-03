#!/bin/bash --login
#SBATCH --job-name=finetune
#SBATCH --output=logs/100kDataExp/out_without_finetune.txt
#SBATCH --error=logs/100kDataExp/err_without_finetune.txt
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=13g
#SBATCH --gres=gpu:1
#SBATCH -t 0-6:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Running Finetuning on Mcrae - fine_tune_mcrae_data_config' 
python3 fine_tune.py --config_file configs/fine_tune/without_finetune_mcrae_data_config.json

echo 'finished!'