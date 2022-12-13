#!/bin/bash --login

#SBATCH --job-name=step5

#SBATCH --output=logs/get_con_pro_embeddings/out_5_get_embeds_predict_property_cnet_premium.txt
#SBATCH --error=logs/get_con_pro_embeddings/err_5_get_embeds_predict_property_cnet_premium.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10gs
#SBATCH --gres=gpu:1

#SBATCH -t 0-02:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_sim_props.py --config_file configs/generate_embeddings/5_get_embeds_predict_property_cnet_premium.json

echo 'finished!'