# 
exp_desc=tune-$1 # the name of hyperparameter

cd /tmp2/cwlin/explainable_credit/explainable_credit
cp -r experiments experiments-$exp_desc

cd /tmp2/cwlin/explainable_credit/explainable_credit/experiments-$exp_desc

find . -name train.log -delete
find . -name checkpoint -delete
find . -name pred.csv -delete
find . -name fixed_pred.csv -delete
find . -name metrics*.json -delete
find . -name test_metrics*.json -delete
find . -name results_pt.csv -delete

echo done