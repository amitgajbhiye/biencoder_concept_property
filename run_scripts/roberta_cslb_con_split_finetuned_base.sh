#!/bin/bash --login

#SBATCH --job-name=cs_RBF

#SBATCH --output=logs/cslb_fine_tuned_100k_logs/out_con_split_roberta_base_cslb_finetuned.txt
#SBATCH --error=logs/cslb_fine_tuned_100k_logs/err_con_split_roberta_base_cslb_finetuned.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-20:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'ROBERTa Base - CSLB Fine tuned Training On configs/fine_tune/cslb_con_split_roberta_base_fine_tuned_config.json' 
python3 fine_tune.py --config_file configs/fine_tune/cslb_con_split_roberta_base_fine_tuned_config.json

echo 'finished!'