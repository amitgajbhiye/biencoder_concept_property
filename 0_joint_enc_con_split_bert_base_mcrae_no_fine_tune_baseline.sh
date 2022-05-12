#!/bin/bash --login

#SBATCH --job-name=JmcCSbb

#SBATCH --output=logs/mcrae_bert_base_baseline_log/out_jont_enc_mcrae_con_split_bert_base_baseline.txt
#SBATCH --error=logs/mcrae_bert_base_baseline_log/err_jont_enc_mcrae_con_split_bert_base_baseline.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=5g
#SBATCH --gres=gpu:1

#SBATCH -t 0-01:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 fine_tune.py --config_file 0_joint_enc_con_split_bert_base_mcrae_no_fine_tune_baseline.sh

echo 'finished!'