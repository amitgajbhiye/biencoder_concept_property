#!/bin/bash --login
#SBATCH --job-name=hd_nn

#SBATCH --output=logs/nn_analysis/out_bert_base_mcrae_train_con_prope_mbeddngs_nn_analysis.txt
#SBATCH --error=logs/nn_analysis/err_bert_base_mcrae_train_con_prope_mbeddngs_nn_analysis.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-1:30:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv


python3 nearest_neighbour_analysis.py

echo finished!