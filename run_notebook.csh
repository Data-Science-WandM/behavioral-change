#!/bin/bash
#SBATCH -J au   # Job name
#SBATCH --output=/sciclone/data10/iahewababarand/behavior_change/logs/automation_analysis_output.log        # Output file
#SBATCH --error=/sciclone/data10/iahewababarand/behavior_change/logs/automation_analysis_error.log       # Error file
#SBATCH -N 1                   # Number of nodes
#SBATCH -n 1                   # Number of tasks/cores per node
# SBATCH --gres=gpu:1           # Request 1 GPU
# SBATCH --gpus=1
#SBATCH -t 48:00:00            # Runtime in hh:mm:ss
#SBATCH --mem=30G

# Run your Python script
python -m src.index --task fox8_analyzer --config config/fox8_config.yaml  > logs/fox8_analysis.txt
# python -m src.index --task fox8_analyzer --config config/fox8_config.yaml
# python -m src.index --task infoOps_analyzer --config config/infoOps_config.yaml