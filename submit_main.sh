#!/bin/bash
#$ -cwd
#$ -pe nproc 32
#$ -N pathcalc
#$ -l mem=2000M
#$ -l h_rt=24:00:00
#$ -e error.out
#$ -o output.out

echo "Running code in $(pwd)"

# >>> Conda environment setup <<<
source /home/sgarg/miniconda3/etc/profile.d/conda.sh
conda activate base

source /usr/local/run/gromacs-2018.6-plumed-2.5.1-sse41/bin/GMXRC

python3 -u parallelmain.py > out.txt 2>&1

echo "Finished code in $(pwd)"
