#!/bin/bash --login

#SBATCH --job-name=JmcPSbb

#SBATCH --output=logs/joint_enc_logs/out_jont_enc_mcrae_con_prop_split_bert_base_baseline.txt
#SBATCH --error=logs/joint_enc_logs/err_jont_enc_mcrae_con_prop_split_bert_base_baseline.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=5g
#SBATCH --gres=gpu:1

#SBATCH -t 0-8:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 fine_tune.py --config_file configs/fine_tune/joint_encs/mcrae/con_prop_split_bert_base_without_fine_tune_baseline_config.json

echo 'finished!'