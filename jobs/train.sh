#!/bin/bash


N_GPU=1
N_CPU=16
TIME=24:00

bsub -J "train_muse_morphose" -o lsf.train_muse_morphose_%J -n $N_CPU -W "$TIME" -R "rusage[mem=4096,scratch=5000,ngpus_excl_p=$N_GPU]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
module load gcc/6.3.0 python_gpu/3.8.5 cuda/11.1.1 libsndfile/1.0.28 fluidsynth/1.1.10 eth_proxy
# Make sure that local packages take precedence over preinstalled packages
export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:$PYTHONPATH

python "$HOME/MuseMorphose/train.py" "$HOME/MuseMorphose/config/euler.yaml"
ENDBSUB