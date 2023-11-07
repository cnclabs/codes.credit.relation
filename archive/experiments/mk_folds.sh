#!/bin/bash

path=$1
typ=$2

mkdir ${path}/${typ}_fold_01
mkdir ${path}/${typ}_fold_01/last_weights
mkdir ${path}/${typ}_fold_02
mkdir ${path}/${typ}_fold_02/last_weights
mkdir ${path}/${typ}_fold_03
mkdir ${path}/${typ}_fold_03/last_weights
mkdir ${path}/${typ}_fold_04
mkdir ${path}/${typ}_fold_04/last_weights
mkdir ${path}/${typ}_fold_05
mkdir ${path}/${typ}_fold_05/last_weights
mkdir ${path}/${typ}_fold_06
mkdir ${path}/${typ}_fold_06/last_weights
mkdir ${path}/${typ}_fold_07
mkdir ${path}/${typ}_fold_07/last_weights
mkdir ${path}/${typ}_fold_08
mkdir ${path}/${typ}_fold_08/last_weights
mkdir ${path}/${typ}_fold_09
mkdir ${path}/${typ}_fold_09/last_weights
mkdir ${path}/${typ}_fold_10
mkdir ${path}/${typ}_fold_10/last_weights
mkdir ${path}/${typ}_fold_11
mkdir ${path}/${typ}_fold_11/last_weights
mkdir ${path}/${typ}_fold_12
mkdir ${path}/${typ}_fold_12/last_weights
mkdir ${path}/${typ}_fold_13
mkdir ${path}/${typ}_fold_13/last_weights

touch .gitignore

cp ${path}/params.json ${path}/${typ}_fold_01
cp .gitignore ${path}/${typ}_fold_01/last_weights
cp ${path}/params.json ${path}/${typ}_fold_02
cp .gitignore ${path}/${typ}_fold_02/last_weights
cp ${path}/params.json ${path}/${typ}_fold_03
cp .gitignore ${path}/${typ}_fold_03/last_weights
cp ${path}/params.json ${path}/${typ}_fold_04
cp .gitignore ${path}/${typ}_fold_04/last_weights
cp ${path}/params.json ${path}/${typ}_fold_05
cp .gitignore ${path}/${typ}_fold_05/last_weights
cp ${path}/params.json ${path}/${typ}_fold_06
cp .gitignore ${path}/${typ}_fold_06/last_weights
cp ${path}/params.json ${path}/${typ}_fold_07
cp .gitignore ${path}/${typ}_fold_07/last_weights
cp ${path}/params.json ${path}/${typ}_fold_08
cp .gitignore ${path}/${typ}_fold_08/last_weights
cp ${path}/params.json ${path}/${typ}_fold_09
cp .gitignore ${path}/${typ}_fold_09/last_weights
cp ${path}/params.json ${path}/${typ}_fold_10
cp .gitignore ${path}/${typ}_fold_10/last_weights
cp ${path}/params.json ${path}/${typ}_fold_11
cp .gitignore ${path}/${typ}_fold_11/last_weights
cp ${path}/params.json ${path}/${typ}_fold_12
cp .gitignore ${path}/${typ}_fold_12/last_weights
cp ${path}/params.json ${path}/${typ}_fold_13
cp .gitignore ${path}/${typ}_fold_13/last_weights

rm .gitignore
