#!/bin/bash


N_GPU=1
N_CPU=4
TIME=24:00

CKPT="$SCRATCH/MuseMorphose/ckpt/enc_dec_12L-16_bars-seqlen_512/params/step_97650-RC_0.845-KL_0.242-model.pt"
OUT_FILE="$HOME/MuseMorphose/ppl.csv"
N_PIECES=1024

bsub -J "muse_morphose_ppl" -o lsf.ppl_%J -n $N_CPU -W "$TIME" -R "rusage[mem=4096,scratch=5000,ngpus_excl_p=$N_GPU]" -R "select[gpu_mtotal0>=10240]" <<ENDBSUB
module load gcc/6.3.0 python_gpu/3.8.5 cuda/11.1.1 libsndfile/1.0.28 fluidsynth/1.1.10 eth_proxy
# Make sure that local packages take precedence over preinstalled packages
export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages:$PYTHONPATH

python "$HOME/MuseMorphose/perplexity.py" "$HOME/MuseMorphose/config/euler.yaml" "$CKPT" "$OUT_FILE" $N_PIECES
ENDBSUB