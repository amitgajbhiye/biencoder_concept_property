#!/bin/bash --login

#SBATCH --job-name=a

#SBATCH --output=logs/cslb_prop_split_100k_fine_tuned_logs/out_cslb_ft_mscg_prefix_adj_8k_bert_base.txt
#SBATCH --error=logs/cslb_prop_split_100k_fine_tuned_logs/err_cslb_ft_mscg_prefix_adj_8k_bert_base.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=12g
#SBATCH --gres=gpu:1

#SBATCH -t 0-15:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Bert Base - Training on configs/fine_tune/cslb_pcv_prefix_adjectives_8k_mscg.json' 
python3 fine_tune.py --config_file configs/fine_tune/cslb_pcv_prefix_adjectives_8k_mscg.json

echo 'finished!'