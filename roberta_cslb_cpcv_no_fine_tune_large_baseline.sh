#!/bin/bash --login

#SBATCH --job-name=cp_RL

#SBATCH --output=logs/cslb_fine_tuned_100k_logs/out_cslb_con_prop_split_roberta_large_no_finetune_baseline.txt
#SBATCH --error=logs/cslb_fine_tuned_100k_logs/err_cslb_con_prop_split_roberta_large_no_finetune_baseline.txt

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

echo 'Roberta Large Con-Prop Split Roberta Large Baseline' - 
echo 'Training On configs/fine_tune/cslb_cpcv_roberta_large_without_fine_tune_data_config.json'
python3 fine_tune.py --config_file configs/fine_tune/cslb_cpcv_roberta_large_without_fine_tune_data_config.json

echo 'finished!'