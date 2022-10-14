#!/bin/bash --login

#SBATCH --job-name=getProps

#SBATCH --output=logs/get_con_pro_embeddings/out_bert_base_gkb_cnet_trained_model_mcrae_concept_similar_properties.txt
#SBATCH --error=logs/get_con_pro_embeddings/err_bert_base_gkb_cnet_trained_model_mcrae_concept_similar_properties.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10gs
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_similar_properties.py --config_file configs/generate_embeddings/get_concept_similar_properties_config.json

echo 'finished!'