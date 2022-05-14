#!/bin/bash --login
#SBATCH --job-name=hd_nn

#SBATCH --output=logs/nn_analysis/out_property_embeddngs_hd_data_nn_analysis.txt
#SBATCH --error=logs/nn_analysis/err_property_embeddngs_hd_data_nn_analysis.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1

#SBATCH -t 0-0:30:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running HD Data NN Analysis experiment...'

python3 nearest_neighbour_analysis.py

echo finished!