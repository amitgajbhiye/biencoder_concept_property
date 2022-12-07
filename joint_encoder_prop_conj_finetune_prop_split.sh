#!/bin/bash --login

#SBATCH --job-name=FtPSMc

#SBATCH --output=logs/joint_enc_logs/out_joint_encoder_prop_conj_prop_split_mcrae_finetune.txt
#SBATCH --error=logs/joint_enc_logs/err_joint_encoder_prop_conj_prop_split_mcrae_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=14g
#SBATCH --gres=gpu:1

#SBATCH -t 0-02:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/joint_encoder_property_conjuction.py --finetune --cv_type="property_split"

echo 'finished!'