#!/bin/bash

for i in 01 06 12;
do
    ./mk_folds.sh mlp${i} time
    ./mk_folds.sh gru${i} time
    ./mk_folds.sh lstm${i} time
    ./mk_folds.sh mlp${i}_index index
    ./mk_folds.sh gru${i}_index index
    ./mk_folds.sh lstm${i}_index index
done
