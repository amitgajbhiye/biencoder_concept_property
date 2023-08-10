#!/bin/bash --login

#SBATCH --job-name=getEmb

#SBATCH --output=logs/out_get_embeds.txt
#SBATCH --error=logs/err_get_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p dev
#SBATCH --mem=8G
##SBATCH --gres=gpu:1

#SBATCH -t 0-00:40:00

conda activate venv

# python3 get_embedding.py --config_file configs/generate_embeddings/get_concept_property_embeddings.json

python3 get_embedding.py --config_file configs/generate_embeddings/get_concept_property_embeddings_bert_large.json

echo 'finished!'