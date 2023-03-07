# bash

# Remove existed folder
if [ -d "../data/interim/sample" ]
then
    rm -rf ../data/interim/sample
fi

# Recreate folder
mkdir ../data/interim/sample

# Copy file
cp ../data/interim/data_cpd.csv ../data/interim/sample/data_cpd.csv
cp ../data/interim/data_fea.csv ../data/interim/sample/data_fea.csv 
cp ../data/interim/label_cum.csv ../data/interim/sample/label_cum.csv
cp ../data/interim/label_for.csv ../data/interim/sample/label_for.csv

# Sampling merged.csv
python sampling.py --file_path=../data/interim/merged.csv --output_dir=../data/interim/sample/ --mode=random --sample_time=500


# Remove existed folder
if [ -d "../data/processed/sample" ]
then
    rm -rf ../data/processed/sample
fi

# Recreate folder
mkdir ../data/processed/sample
mkdir ../data/processed/sample/8_labels_index
mkdir ../data/processed/sample/8_labels_time

for i in 01 06 12
do 
    mkdir ../data/processed/sample/8_labels_index/len_${i}
    mkdir ../data/processed/sample/8_labels_time/len_${i}
done

# Make δ RNN Units
for window_size in 01 06 12
do
    python make_sequence.py --path=../data/interim/sample/ --window=${window_size} --step=1
done

# Cross Section & Cross Time
for type in 'index' 'time'
do
    for i in 01 06 12
    do
        python data_splitter.py --input_file=../data/interim/sample/seq_w_${i}_s_01.csv --output_path=../data/processed/sample/8_labels_${type}/len_${i} --mode=${type}
    done
done