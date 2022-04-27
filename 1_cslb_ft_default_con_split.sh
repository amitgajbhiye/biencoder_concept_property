#!/bin/bash --login

#SBATCH --job-name=blmag

#SBATCH --output=logs/cslb_con_split_100k_fine_tuned_logs/test_out_cslb_ft_bert_large_mscg_plus_prefix_adjective_plus_gkb_100k.txt
#SBATCH --error=logs/cslb_con_split_100k_fine_tuned_logs/test_err_cslb_ft_bert_large_mscg_plus_prefix_adjective_plus_gkb_100k.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=15g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Bert Large - Training On configs/fine_tune/ft_cslb_bert_large_fine_tune_mcrae_data_config.json' 
python3 fine_tune.py --config_file configs/fine_tune/ft_cslb_bert_large_fine_tune_mcrae_data_config.json

echo 'finished!'