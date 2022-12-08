#!/bin/bash --login

#SBATCH --job-name=ConPropMcRaeEmbedsings

#SBATCH --output=logs/get_con_pro_embeddings/out_mcrae_fine_tune_property_split_property_comjuction_folds_data.txt
#SBATCH --error=logs/get_con_pro_embeddings/err_mcrae_fine_tune_property_split_property_comjuction_folds_data.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
##SBATCH -p gpu_v100,gpu
#SBATCH --mem=10gs
#SBATCH --gres=gpu:1

#SBATCH -t 0-03:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_similar_properties.py --config_file configs/generate_embeddings/get_predict_prop_similar_vocab_properties_from_gkb_cnet_cnet_has_prop_pretrained_bb_model_config.json

echo 'finished!'