#!/bin/bash --login

#SBATCH --job-name=JE

#SBATCH --output=logs/joint_enc_logs/out_joint_encoder.txt
#SBATCH --error=logs/joint_enc_logs/err_joint_encoder.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=12gs
#SBATCH --gres=gpu:1

#SBATCH -t 1-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 joint_encoder.py

echo 'finished!'