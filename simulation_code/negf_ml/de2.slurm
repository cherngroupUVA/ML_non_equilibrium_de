#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --partition=standard
#SBATCH --time=7-00:00:00
#SBATCH --mem=24000
#SBATCH --account=mott-phys

module purge
module load gcc/7.1.0
module load python/3.6.8
module load armadillo

python main.py

