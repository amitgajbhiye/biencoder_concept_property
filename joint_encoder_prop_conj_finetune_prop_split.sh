#!/bin/bash --login

#SBATCH --job-name=FtPSMc

#SBATCH --output=logs/je_logs/out_new_joint_encoder_prop_conj_prop_split_mcrae_finetune.txt
#SBATCH --error=logs/je_logs/err_new_joint_encoder_prop_conj_prop_split_mcrae_finetune.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=9g
#SBATCH --gres=gpu:1

#SBATCH -t 0-10:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/joint_encoder_property_conjuction.py --finetune --cv_type="property_split"

echo 'finished !!'