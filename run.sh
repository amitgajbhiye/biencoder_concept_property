#!/bin/bash --login
#SBATCH --job-name=biencoder
#SBATCH --output=logs/bce_out.file
#SBATCH --error=logs/bce_err.file
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p dev
#SBATCH --mem=10g
##SBATCH --gres=gpu:1
##SBATCH --qos=gpu7d
#SBATCH -t 0-01:00:00

echo 'This script is running on:'
hostname

#clush -w $SLURM_NODELIST "sudo /apps/slurm/gpuset_0_shared"

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 run_model.py --config_file configs/default_config.json

echo finished!
