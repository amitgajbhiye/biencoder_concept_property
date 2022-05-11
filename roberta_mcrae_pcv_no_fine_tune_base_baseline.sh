#!/bin/bash --login

#SBATCH --job-name=mcPSrb

#SBATCH --output=logs/mcrae_roberta_logs/out_mcrae_prop_split_roberta_base_baseline.txt
#SBATCH --error=logs/mcrae_roberta_logs/err_mcrae_prop_split_roberta_base_baseline.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 fine_tune.py --config_file configs/fine_tune/mcrae_pcv_roberta_base_without_fine_tune_mcrae_data_config.json

echo 'finished!'