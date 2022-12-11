#!/bin/bash --login

#SBATCH --job-name=PropVocabEmbeds

#SBATCH --output=logs/get_con_pro_embeddings/out_get_embeds_prop_vocab_from_gkb_cnet_cnet_has_prop_pretrained_bienc_bb_model.txt
#SBATCH --error=logs/get_con_pro_embeddings/err_get_embeds_prop_vocab_from_gkb_cnet_cnet_has_prop_pretrained_bienc_bb_model.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=5g


#SBATCH -t 0-02:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_similar_properties.py --config_file configs/generate_embeddings/get_embeds_prop_vocab_from_gkb_cnet_cnet_has_prop_pretrained_bienc_bb_model_config.json

echo 'finished!'