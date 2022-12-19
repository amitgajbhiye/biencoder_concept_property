#!/bin/bash --login

#SBATCH --job-name=JEcpPre

#SBATCH --output=logs/je_logs/out_je_pretrain_con_prop_cnet_premium_20negdata.txt
#SBATCH --error=logs/je_logs/err_je_pretrain_con_prop_cnet_premium_20negdata.txt

#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=10g
#SBATCH --gres=gpu:1
#SBATCH --qos="gpu7d"

#SBATCH -t 5-00:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

python3 model/joint_encoder_concept_property.py --pretrain

echo 'finished!'