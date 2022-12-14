#!/bin/bash --login

#SBATCH --job-name=JEPropConjPreTrain

#SBATCH --output=logs/je_logs/out_8_je_prop_conj_pretraining_cnet_premium_data.txt
#SBATCH --error=logs/je_logs/err_8_je_prop_conj_pretraining_cnet_premium_data.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=14g
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/joint_encoder_property_conjuction.py --pretrain

echo 'finished!'