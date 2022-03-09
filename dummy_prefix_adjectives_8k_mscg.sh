#!/bin/bash --login
#SBATCH --job-name=a

#SBATCH --output=logs/100k_property_split_logs/out_dummy_prefix_adj.txt
#SBATCH --error=logs/100k_property_split_logs/err_dummy_prefix_adj.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5

#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1
#SBATCH -t 0-10:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Property CV bert base fine tune on McRae train set config_file - configs/fine_tune/pcv_prefix_adjectives_8k_mscg.json' 

python3 fine_tune.py --config_file configs/fine_tune/pcv_prefix_adjectives_8k_mscg.json

echo 'finished!'