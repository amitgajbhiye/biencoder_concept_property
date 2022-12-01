#!/bin/bash --login

#SBATCH --job-name=JEPreTrain

#SBATCH --output=logs/joint_enc_logs/out_dummy_data_joint_encoder_prop_conj_pretraining.txt
#SBATCH --error=logs/joint_enc_logs/err_dummy_data_joint_encoder_prop_conj_pretraining.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/prop_conjuction_joint_encoder.py --pretrain

echo 'finished!'