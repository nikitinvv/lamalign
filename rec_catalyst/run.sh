#!/bin/bash
#SBATCH -t 8:00:00
#SBATCH -J laminography
#SBATCH -p v100
##SBATCH --exclude gn1
#SBATCH -c 42
#SBATCH --mem 160G

module add GCC/8.3.0  GCCcore/8.3.0  CUDA/10.1.243

CUDA_VISIBLE_DEVICES=0 python rec_align_matlab.py 
#CUDA_VISIBLE_DEVICES=1 python rec_align_reg.py 4e-9 &
#wait

