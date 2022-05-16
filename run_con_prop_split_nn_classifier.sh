#!/bin/bash --login
#SBATCH --job-name=CP_NNcls

#SBATCH --output=logs/nn_analysis/out_bert_base_mcrae_con_prop_split_nn_3_classifier.txt
#SBATCH --error=logs/nn_analysis/err_bert_base_mcrae_con_prop_split_nn_3_classifier.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=6g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 con_prop_split_classifier_nn.py

echo finished!