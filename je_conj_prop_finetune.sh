#!/bin/bash --login

#SBATCH --job-name=CPMr5FT

#SBATCH --output=logs/redo_prop_conj_exp/out_mcrae_con_prop_propsplit_fintune_5neg_pretrain_model.txt
#SBATCH --error=logs/redo_prop_conj_exp/err_mcrae_con_prop_propsplit_fintune_5neg_pretrain_model.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=9g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/je_con_prop.py --finetune --cv_type="property_split"

echo 'finished !!'