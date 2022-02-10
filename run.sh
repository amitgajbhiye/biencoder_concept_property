#!/bin/bash --login
#SBATCH --job-name=8bs70kcntx4
#SBATCH --output=logs/70k_8bs_cntx_4_out.file
#SBATCH --error=logs/70k_8bs_cntx_4_err.file
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100,gpu
#SBATCH --mem=15g
#SBATCH --gres=gpu:1
#SBATCH -t 0-4:00:00

echo 'This script is running on:'
hostname

#clush -w $SLURM_NODELIST "sudo /apps/slurm/gpuset_0_shared"

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

python3 run_model.py --config_file configs/context_4_bs_8_config.json

echo finished!
