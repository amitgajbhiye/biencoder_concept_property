#!/bin/bash --login

#SBATCH --job-name=getLogits

#SBATCH --output=logs/joint_enc_logs/out_pretrained_joint_encoder_con_similar_prop_filtering_threshold_50.txt
#SBATCH --error=logs/joint_enc_logs/err_pretrained_joint_encoder_con_similar_prop_filtering_threshold_50.txt

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