#!/bin/bash
#
#SBATCH --job-name=transformer
#SBATCH --output=output.txt
#SBATCH --ntasks=1
#SBATCH --partition=students
#SBATCH --gres=gpu:mem11g:1
#SBATCH --mem=16000
#SBATCH --mail-user=anhtu@cl.uni-heidelberg.de
#SBATCH --mail-type=ALL

srun /workspace/students/anhtu/transformers-text-recognition/bash/run_training.sh