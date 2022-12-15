#!/bin/bash --login

#SBATCH --job-name=getLogits

#SBATCH --output=logs/get_con_pro_embeddings/mcrae_logs/out_4_filter_con_similar_props.txt
#SBATCH --error=logs/get_con_pro_embeddings/mcrae_logs/err_4_filter_con_similar_props.txt

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

python3 joint_enc_filter_similar_props.py

echo 'finished!'