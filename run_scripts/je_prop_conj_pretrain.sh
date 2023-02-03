#!/bin/bash --login

#SBATCH --job-name=PC5neg

#SBATCH --output=logs/je_logs/out_8_je_prop_conj_pretrained_cnetp_je_5neg_cnetp_filtered_props.txt
#SBATCH --error=logs/je_logs/err_8_je_prop_conj_pretrained_cnetp_je_5neg_cnetp_filtered_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=14g
#SBATCH --gres=gpu:1

#SBATCH --qos="gpu7d"
#SBATCH -t 5-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_property_conjuction.py --pretrain

echo 'finished!'