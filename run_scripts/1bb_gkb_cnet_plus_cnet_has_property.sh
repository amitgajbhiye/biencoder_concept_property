#!/bin/bash --login

#SBATCH --job-name=gkbCnetHasProp

#SBATCH --output=logs/gkb_source_analysis/out_bb_gkb_cnet_plus_cnet_has_property.txt
#SBATCH --error=logs/gkb_source_analysis/err_bb_gkb_cnet_plus_cnet_has_property.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

##SBATCH --qos=gpu7d
#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 run_model.py --config_file configs/gkb_source_analysis/bb_gkb_cnet_plus_cnet_has_property_config.json

echo 'finished!'