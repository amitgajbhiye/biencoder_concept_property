#!/bin/bash --login
#SBATCH --job-name=context_1_dot_product
#SBATCH --output=logs/context_1_dot_product_out.file
#SBATCH --error=logs/context_1_dot_product_err.file
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu,gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu7d
#SBATCH -t 0-13:00:00

echo 'This script is running on:'
hostname

#clush -w $SLURM_NODELIST "sudo /apps/slurm/gpuset_0_shared"

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 run_model.py --config_file configs/context_1_config.json

echo finished!
