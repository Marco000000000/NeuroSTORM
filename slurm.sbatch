#!/bin/bash
#SBATCH --job-name=ucla_all
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --open-mode=append

# Evita problemi NCCL
export NCCL_P2P_DISABLE=1

# Esecuzione senza buffering
stdbuf -oL -eL conda run -n neurostorm python -u master_pipeline_miccai.py
