#!/bin/bash

path=$1
typ=$2

mkdir ${path}/${typ}_fold_01
mkdir ${path}/${typ}_fold_02
mkdir ${path}/${typ}_fold_03
mkdir ${path}/${typ}_fold_04
mkdir ${path}/${typ}_fold_05
mkdir ${path}/${typ}_fold_06
mkdir ${path}/${typ}_fold_07
mkdir ${path}/${typ}_fold_08
mkdir ${path}/${typ}_fold_09
mkdir ${path}/${typ}_fold_10
mkdir ${path}/${typ}_fold_11
mkdir ${path}/${typ}_fold_12
mkdir ${path}/${typ}_fold_13

cp ${path}/params.json ${path}/${typ}_fold_01
cp ${path}/params.json ${path}/${typ}_fold_02
cp ${path}/params.json ${path}/${typ}_fold_03
cp ${path}/params.json ${path}/${typ}_fold_04
cp ${path}/params.json ${path}/${typ}_fold_05
cp ${path}/params.json ${path}/${typ}_fold_06
cp ${path}/params.json ${path}/${typ}_fold_07
cp ${path}/params.json ${path}/${typ}_fold_08
cp ${path}/params.json ${path}/${typ}_fold_09
cp ${path}/params.json ${path}/${typ}_fold_10
cp ${path}/params.json ${path}/${typ}_fold_11
cp ${path}/params.json ${path}/${typ}_fold_12
cp ${path}/params.json ${path}/${typ}_fold_13
