#!/bin/bash

# Slurm submission script for nano-gpt-124m
# Based on CRIHAN v 1.00 - Jan 2017

# --- Job Configuration ---

# Job name
#SBATCH -J "nano_gpt_train"

# Batch output and error files (make sure 'logs' directory exists)
#SBATCH --output=logs/o%J.log
#SBATCH --error=logs/e%J.log

# Partition (submission class)
#SBATCH --partition gpu  

# --- Resource Request ---

# Request 1 compute node
#SBATCH --nodes=1

# Request 2 GPUs on that node
#SBATCH --gpus=2
# Request 2 tasks (1 task per GPU)
#SBATCH --tasks-per-node=2

# CPUs per task (4-8 is good for A100 data loading)
#SBATCH --cpus-per-task 4

# Job time (hh:mm:ss) - 15 hours
#SBATCH --time 15:00:00

# Job maximum memory (CPU RAM)
#SBATCH --mem 120gb

# --- A100 GPU Specific Request ---
# This line specifically requests nodes with A100s.
#SBATCH --constraint=a100

# --- Notifications ---
#SBATCH --mail-type ALL
# User e-mail address
#SBATCH --mail-user your.email@example.com

# --- Job Steps ---

echo "----------------------------------------------------"
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "----------------------------------------------------"

# 1. Environment setup
echo "Loading modules..."
module purge
module load aidl/pytorch/2.0.0-cuda11.7

echo "Activating Python virtual environment..."
source ~/venvs/nano-gpt-env-py310/bin/activate

# 2. Run the training
# We use torchrun, which works with Slurm to set up
# the distributed training environment variables.
# --nproc_per_node=2 tells it to launch 2 processes, one for each GPU.
echo "Starting training with torchrun..."

torchrun --nproc_per_node=2 train_gpt2.py

echo "----------------------------------------------------"
echo "Job finished at $(date)"
echo "----------------------------------------------------"