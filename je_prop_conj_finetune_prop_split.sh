#!/bin/bash --login

#SBATCH --job-name=FtPSMc

#SBATCH --output=logs/redo_prop_conj_exp/out_mcrae_epoch_prop_split_finetuned_5neg_50threshold_cnetp_pretrained_model.txt
#SBATCH --error=logs/redo_prop_conj_exp/err_mcrae_prop_split_finetuned_5neg_50threshold_cnetp_pretrained_model.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=9g
#SBATCH --gres=gpu:1

#SBATCH -t 0-08:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_property_conjuction.py --finetune --cv_type="property_split"

echo 'finished !!'