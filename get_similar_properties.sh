#!/bin/bash --login

#SBATCH --job-name=ConPropMcRaeEmbedsings

#SBATCH --output=logs/get_con_pro_embeddings/out_bb_gkb_cnet_pretrained_model_generated_mcrae_concept_and_property_embeddings.txt
#SBATCH --error=logs/get_con_pro_embeddings/err_bb_gkb_cnet_pretrained_model_generated_mcrae_concept_and_property_embeddings.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10gs
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:30:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_similar_properties.py --config_file configs/generate_embeddings/get_mcrae_con_prop_embeds_from_gkb_cnet_cnet_has_prop_pretrained_bb_model_config.json

echo 'finished!'