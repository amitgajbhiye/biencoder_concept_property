#!/bin/bash --login

#SBATCH --job-name=step6

#SBATCH --output=logs/redo_prop_conj_exp/out_6_get_predict_property_similiar_props_20neg_50threshold.txt
#SBATCH --error=logs/redo_prop_conj_exp/err_6_get_predict_property_similiar_props_20neg_50threshold.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

##SBATCH -p gpu_v100,gpu
##SBATCH --mem=8g
##SBATCH --gres=gpu:1

#SBATCH -p compute
#SBATCH -t 0-12:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/redo_con_prop_exp/6_get_predict_property_similiar_props_20neg_50_threshold.json

echo 'finished!'