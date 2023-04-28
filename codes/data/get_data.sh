#!/bin/bash

echo 'Copy data from server cfda4:/tmp2/cywu/default_cumulative/data/processed/'
cp -r /tmp2/cywu/default_cumulative/data/processed/. ./all/

python3 ../build_sampling/preprocess/overtime_revise.py --ROOT ./all
python3 ../build_sampling/preprocess/get_all_company_ids.py --ROOT ./all