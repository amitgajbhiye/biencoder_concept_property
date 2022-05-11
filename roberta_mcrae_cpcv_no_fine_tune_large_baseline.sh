#!/bin/bash --login

#SBATCH --job-name=mcCPrl

#SBATCH --output=logs/mcrae_roberta_logs/out_mcrae_con_prop_split_bert_large_no_finetune_baseline.txt
#SBATCH --error=logs/mcrae_roberta_logs/err_mcrae_con_prop_split_bert_large_no_finetune_baseline.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=15g
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 fine_tune.py --config_file configs/fine_tune/mcrae_cpcv_roberta_large_without_fine_tune_data_config.json

echo 'finished!'