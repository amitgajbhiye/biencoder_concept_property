#!/bin/bash --login

#SBATCH --job-name=JEdataPrep

#SBATCH --output=logs/joint_encoder_pretraining_data_prep/out_joint_encoder_pretraining_data_prep.txt
#SBATCH --error=logs/joint_encoder_pretraining_data_prep/err_joint_encoder_pretraining_data_prep.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p highmem
#SBATCH --mem=5g

#SBATCH -t 0-06:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 data/joint_encoder_pretraining_data_prep.py

echo 'finished!'