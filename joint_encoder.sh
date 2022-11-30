#!/bin/bash --login

#SBATCH --job-name=JE

#SBATCH --output=logs/joint_enc_logs/out_pretrain_joint_enc_step2_on_gkbcnet_cnethasprop.txt
#SBATCH --error=logs/joint_enc_logs/err_pretrain_joint_enc_step2_on_gkbcnet_cnethasprop.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --qos=gpu7d
#SBATCH --mem=14gs
#SBATCH --gres=gpu:1

#SBATCH -t 5-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 joint_encoder.py --pretrain

echo 'finished!'