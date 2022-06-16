#!/bin/bash

for i in 01 06 12;
do
    mkdir mlp${i}
    mkdir gru${i}
    mkdir lstm${i}
    mkdir mlp${i}_index
    mkdir gru${i}_index
    mkdir lstm${i}_index
done
