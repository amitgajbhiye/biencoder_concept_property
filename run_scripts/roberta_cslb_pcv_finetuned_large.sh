#!/bin/bash --login

#SBATCH --job-name=ps_RLF

#SBATCH --output=logs/cslb_fine_tuned_100k_logs/out_prop_split_roberta_large_cslb_finetuned.txt
#SBATCH --error=logs/cslb_fine_tuned_100k_logs/err_prop_split_roberta_large_cslb_finetuned.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1

#SBATCH --qos="gpu7d"
#SBATCH -t 3-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Roberta Large - Training On configs/fine_tune/cslb_pcv_roberta_large_fine_tuned_cslb_data_config.json' 
python3 fine_tune.py --config_file configs/fine_tune/cslb_pcv_roberta_large_fine_tuned_cslb_data_config.json

echo 'finished!'