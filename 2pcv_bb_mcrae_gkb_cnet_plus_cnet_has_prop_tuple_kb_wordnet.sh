#!/bin/bash --login
#SBATCH --job-name=pCNetTkWn

#SBATCH --output=logs/mcrae_fine_tune_gkb_source_analysis/out_prop_split_bb_mcrae_pcv_bb_gkb_cnet_plus_cnet_has_prop_tuple_kb_wordnet.txt
#SBATCH --error=logs/mcrae_fine_tune_gkb_source_analysis/err_prop_split_bb_mcrae_pcv_bb_gkb_cnet_plus_cnet_has_prop_tuple_kb_wordnet.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5

#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1
#SBATCH -t 0-12:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 fine_tune.py --config_file configs/fine_tune/gkb_source_analysis/bb_mcrae_pcv_bb_gkb_cnet_plus_cnet_has_prop_tuple_kb_wordnet_config.json

echo 'finished!'