#!/bin/bash --login

#SBATCH --job-name=JcPSbb

#SBATCH --output=logs/joint_enc_logs/out_joint_enc_cslb_prop_split_bert_base_baseline.txt
#SBATCH --error=logs/joint_enc_logs/err_joint_enc_cslb_prop_split_bert_base_baseline.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=5g
#SBATCH --gres=gpu:1

#SBATCH -t 0-23:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 fine_tune.py --config_file configs/fine_tune/joint_encs/mcrae/cslb_prop_split_bert_base_without_fine_tune_baseline_config.json

echo 'finished!'