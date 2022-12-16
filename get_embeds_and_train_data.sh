#!/bin/bash --login

#SBATCH --job-name=MCstep5

#SBATCH --output=logs/get_con_pro_embeddings/mcrae_logs/out_7_mcrae_get_predict_prop_similar_props.txt
#SBATCH --error=logs/get_con_pro_embeddings/mcrae_logs/err_7_mcrae_get_predict_prop_similar_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=8g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:30:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 get_embeds_and_train_data.py --config_file configs/generate_embeddings/mcrae_data/7_mcrae_get_predict_prop_similar_props.json

echo 'finished!'