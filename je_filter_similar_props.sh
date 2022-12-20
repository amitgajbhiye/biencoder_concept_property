#!/bin/bash --login

#SBATCH --job-name=getLogJE10neg

#SBATCH --output=logs/redo_prop_conj_exp/out_4_je_10neg_filter_con_similar_props.txt
#SBATCH --error=logs/redo_prop_conj_exp/err_4_je_10neg_filter_con_similar_props.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:30:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 je_filter_similar_props.py

echo 'finished!'