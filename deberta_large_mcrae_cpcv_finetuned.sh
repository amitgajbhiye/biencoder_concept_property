#!/bin/bash --login

#SBATCH --job-name=cpmDLf

#SBATCH --output=logs/cslb_deberta_logs/out_mcrae_con_prop_split_deberta_large_finetuned.txt
#SBATCH --error=logs/cslb_deberta_logs/err_mcrae_con_prop_split_deberta_large_finetuned.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1

#SBATCH -t 2-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 fine_tune.py --config_file configs/fine_tune/deberta_large_mcrae_cpcv_fine_tuned_config.json

echo 'finished!'