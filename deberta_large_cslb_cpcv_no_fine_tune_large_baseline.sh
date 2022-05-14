#!/bin/bash --login

#SBATCH --job-name=cp_DL

#SBATCH --output=logs/cslb_deberta_logs/out_cslb_con_prop_split_deberta_large_no_finetune_baseline.txt
#SBATCH --error=logs/cslb_deberta_logs/err_cslb_con_prop_split_deberta_large_no_finetune_baseline.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'


python3 fine_tune.py --config_file configs/fine_tune/deberta_large_cslb_cpcv_without_fine_tune_config.json

echo 'finished!'