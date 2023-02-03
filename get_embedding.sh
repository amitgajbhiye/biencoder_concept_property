#!/bin/bash --login

#SBATCH --job-name=getEmb

#SBATCH --output=logs/out_get_embeds.txt
#SBATCH --error=logs/err_get_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10gs
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00


# module load anaconda/2020.02

# . activate

conda activate venv

python3 get_embedding.py --config_file configs/generate_embeddings/get_concept_property_embeddings.json

echo 'finished!'