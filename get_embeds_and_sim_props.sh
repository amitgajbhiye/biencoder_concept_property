#!/bin/bash --login

#SBATCH --job-name=step2

#SBATCH --output=logs/get_con_pro_embeddings/out_2_get_embeds_prop_vocab_500k_mscg_embeds.txt
#SBATCH --error=logs/get_con_pro_embeddings/err_2_get_embeds_prop_vocab_500k_mscg_embeds.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p compute
#SBATCH --mem=5g

#SBATCH -t 0-02:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_sim_props.py --config_file configs/generate_embeddings/2_get_embeds_prop_vocab.json

echo 'finished!'