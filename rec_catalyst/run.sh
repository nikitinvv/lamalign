#!/bin/bash
#SBATCH -t 4:00:00
#SBATCH -J laminography
#SBATCH -p v100
##SBATCH --exclude gn1
#SBATCH -c 64
#SBATCH --mem 160G

module add GCC/8.3.0  GCCcore/8.3.0  CUDA/10.1.243

python rec_align.py 

