#!/bin/bash --login
#SBATCH --job-name=cntx_exp_
#SBATCH --output=logs/70k_cntx_batch_exp_out.file
#SBATCH --error=logs/70k_cntx_batch_exp_err.file
#SBATCH --tasks-per-node=5
#SBATCH --ntasks=5
#SBATCH -A scw1858

#SBATCH -p gpu_v100
#SBATCH --mem=15g
#SBATCH --gres=gpu:1
#SBATCH -t 1-10:00:00

echo 'This script is running on:'
hostname

module load anaconda/2020.02

. activate

conda activate venv

echo 'Running experiment...'

echo 'Running Context 1 Mean Strategy' 
python3 run_model.py --config_file configs/mean_vector_strategy/context_1_mean.json

echo

echo 'Running Context 2 Mean Strategy' 
python3 run_model.py --config_file configs/mean_vector_strategy/context_2_mean.json

echo

echo 'Running Context 3 Mean Strategy' 
python3 run_model.py --config_file configs/mean_vector_strategy/context_3_mean.json

echo

echo 'Running Context 4 Mean Strategy' 
python3 run_model.py --config_file configs/mean_vector_strategy/context_4_mean.json

echo

echo 'Running Context 5 Mean Strategy' 
python3 run_model.py --config_file configs/mean_vector_strategy/context_5_mean.json

echo

echo '***************************************************'
echo '***** Running Mask Token Strategy Experiments******'
echo '***************************************************'

echo

echo 'Running Context 6 Mask Strategy' 
python3 run_model.py --config_file configs/mask_vector_strategy/context_6_msk.json

echo

echo 'Running Context 7 Mask Strategy' 
python3 run_model.py --config_file configs/mask_vector_strategy/context_7_msk.json

echo

echo 'Running Context 8 Mask Strategy' 
python3 run_model.py --config_file configs/mask_vector_strategy/context_8_msk.json

echo

echo 'Running Context 9 Mask Strategy' 
python3 run_model.py --config_file configs/mask_vector_strategy/context_9_msk.json

echo

echo 'Running Context 10 Mask Strategy' 
python3 run_model.py --config_file configs/mask_vector_strategy/context_10_msk.json

echo

echo 'Running Context 11 Mask Strategy' 
python3 run_model.py --config_file configs/mask_vector_strategy/context_11_msk.json

echo

echo 'Running Context 12 Mask Strategy' 
python3 run_model.py --config_file configs/mask_vector_strategy/context_12_msk.json

echo 'finished!'
