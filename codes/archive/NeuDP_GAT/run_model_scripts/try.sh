#!/bin/bash

CSV_FILE='min_rmsne_GAT.csv'

device=$1
lstm_num_units=$2
fold_start=$3
fold_end=$4

experiment_type='time'
cluster_setting='industry'
n_cluster=14

# Read the CSV file line by line
for fold in $(seq $fold_end -1 $fold_start); do
    while IFS=, read -r lstm intra inter lr wd
    do
        # Skip the header 
        if [[ $lstm == "lstm" ]]; then
            continue
        fi

        # Check if the current line's lstm value matches the targe
        if [[ $lstm == $lstm_num_units ]]; then
            echo $fold, $lstm, $intra, $inter, $lr, $wd
            # bash run_GAT_fold_03.sh $device $experiment_type $cluster_setting $n_cluster $lstm \
            #                         $intra $inter $lr $wd $fold_start $fold_end &&
            echo "Done with $lstm, $intra, $inter, $lr, $wd"
        fi


    done < $CSV_FILE
done
