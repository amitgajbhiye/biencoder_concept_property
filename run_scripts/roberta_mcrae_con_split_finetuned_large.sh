#!/bin/bash --login

#SBATCH --job-name=mcCSrlf

#SBATCH --output=logs/mcrae_roberta_logs/out_con_split_roberta_large_mcrae_finetuned.txt
#SBATCH --error=logs/mcrae_roberta_logs/err_con_split_roberta_large_mcrae_finetuned.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1

#SBATCH -t 0-12:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 fine_tune.py --config_file configs/fine_tune/mcrae_con_split_roberta_large_fine_tuned_config.json

echo 'finished!'