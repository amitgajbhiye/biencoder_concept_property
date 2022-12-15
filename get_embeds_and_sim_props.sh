#!/bin/bash --login

#SBATCH --job-name=MCstep5

#SBATCH --output=logs/get_con_pro_embeddings/mcrae_logs/out_5_get_embeds_mcrae_predict_property.txt
#SBATCH --error=logs/get_con_pro_embeddings/mcrae_logs/err_5_get_embeds_mcrae_predict_property.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_sim_props.py --config_file configs/generate_embeddings/mcrae_data/5_get_embeds_mcrae_predict_property_cnet_premium.json

echo 'finished!'