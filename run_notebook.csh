#!/bin/bash
#SBATCH -J au   # Job name
#SBATCH --output=logs/automation_analysis_output.log        # Output file
#SBATCH --error=logs/automation_analysis_error.log       # Error file
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 1                   # Number of tasks/cores per node
# SBATCH --gres=gpu:1           # Request 1 GPU
# SBATCH --gpus=1
#SBATCH -t 48:00:00            # Runtime in hh:mm:ss
#SBATCH --mem=30G

# Run your Python script
# python -m src.index --task retraining_analyzer --config config/retraining_config.yaml  > logs/retraining_analysis.txt
python -m src.index --task fox8_analyzer --config config/fox8_config.yaml > logs/fox8.txt
# python -m src.index --task infoOps_analyzer --config config/infoOps_config.yaml > logs/infoOps.txt
# python -m src.index --task user_analyzer --config config/anaylze_users_config.yaml > logs/anaylze_users_config.txt