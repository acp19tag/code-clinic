#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=tafgreen1@sheffield.ac.uk
#SBATCH --comment=code-clinic-test
#SBATCH --error logs/err.log
#SBATCH --output logs/out.log

bash run.sh