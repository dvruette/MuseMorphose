#!/bin/bash

source scripts/ensure_dataset.sh

for i in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do

bsub -J "precompute_$i" -o lsf.precompute_$i_%J -n 1 -W "24:00" -R "rusage[mem=4096,scratch=5000]" <<ENDBSUB
module load gcc/6.3.0 python_gpu/3.8.5 cuda/11.1.1 libsndfile/1.0.28 fluidsynth/1.1.10 eth_proxy
# Make sure that local packages take precedence over preinstalled packages
export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:$PYTHONPATH

export LMD_DIR="$SCRATCH/lmd/lmd_full/$i"
export REMI_DIR="$SCRATCH/lmd/lmd_remi/$i"
python $HOME/MuseMorphose/precompute_remi.py
ENDBSUB

done