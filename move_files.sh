#!/bin/bash

# mkdir -p $SCRATCH/lmd/lmd_remi/attr_cls/polyph
# mkdir -p $SCRATCH/lmd/lmd_remi/attr_cls/rhythm

# for i in 0 1 2 3 4 5 6 7 8 9 a b c d e f; do
#     mv $SCRATCH/lmd/lmd_remi/$i/attr_cls/polyph $SCRATCH/lmd/lmd_remi/attr_cls/polyph/$i
#     mv $SCRATCH/lmd/lmd_remi/$i/attr_cls/rhythm $SCRATCH/lmd/lmd_remi/attr_cls/rhythm/$i
# done

find $SCRATCH/lmd/lmd_remi -name "*.mid" -type f -exec sh -c 'mv {} "$(dirname {})_$(basename {})"' \;