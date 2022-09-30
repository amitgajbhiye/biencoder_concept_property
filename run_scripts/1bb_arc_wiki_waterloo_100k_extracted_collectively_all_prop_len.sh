#!/bin/bash --login

#SBATCH --job-name=AllPropLen

#SBATCH --output=logs/gkb_source_analysis/out_bb_arc_wiki_waterloo_100k_extracted_collectively_all_prop_len.txt
#SBATCH --error=logs/gkb_source_analysis/err_bb_arc_wiki_waterloo_100k_extracted_collectively_all_prop_len.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH --qos=gpu7d
#SBATCH -t 7-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 run_model.py --config_file configs/gkb_source_analysis/bb_arc_wiki_waterloo_100k_extracted_collectively_all_prop_len.json

echo 'finished!'