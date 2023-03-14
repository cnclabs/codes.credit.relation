import pandas as pd
import os
import argparse
import re

# python3 get_results.py --ROOT /tmp2/cwlin/explainable_credit/explainable_credit --exp_date 0215 --exp_type index --exp_model gru --hyperparameter_name all

parser = argparse.ArgumentParser()
parser.add_argument('--ROOT', default='/tmp2/cwlin/explainable_credit/explainable_credit', type=str, help='experiment date')
parser.add_argument('--exp_date', default='0215', type=str, help='experiment date')
parser.add_argument('--exp_type', default='index', type=str, help='index, time')
parser.add_argument('--exp_model', default='gru', type=str, help='gru, lstm, mlp')
parser.add_argument('--hyperparameter_name', default='all', type=str, help='experiment date')
parser.add_argument('--hyperparameter_value', default='1e-5', type=str, help='experiment date')

args = parser.parse_args()
exp_dir = f'{args.ROOT}/experiments-tune-{args.hyperparameter_name}'
exp_desc = f'revised.index.tune.{args.hyperparameter_name}={args.hyperparameter_value}' if args.hyperparameter_name!= 'all' else f'revised.index.tune.all'
output_dir = '/tmp2/cwlin/explainable_credit/results'

all_results_path = f'/tmp2/cwlin/explainable_credit/results/{args.exp_date}-{exp_desc}'

AR = [f'cap_0{i}' for i in range(1,9)]
RMSNE = [f'recall_0{i}' for i in range(1,9)]

result_dict = dict()
paths = os.listdir(all_results_path)
param_names = ['ws', 'bs', 'patience', 'hidden', 'lr', 'weight_decay', 'layer', 'dropout']
def get_param_value(string):
    ws = re.search(r'ws=(\d+)', string).group(1)
    bs = re.search(r'bs=(\d+)', string).group(1)
    patience = re.search(r'patience=(\d+)', string).group(1)
    hidden = re.search(r'hidden=(\d+)', string).group(1)
    lr = re.search(r'lr=(-?\d+(\.\d+)?(?:e-?\d+)?)', string).group(1)
    weight_decay = re.search(r'weight_decay=(-?\d+(\.\d+)?(?:e-?\d+)?)', string).group(1)
    layer = re.search(r'(\d)layer\.dropout=([\d.]+)', string)
    layer_num = layer.group(1)
    dropout = layer.group(2)
    return ws, bs, patience, hidden, lr, weight_decay, layer_num, dropout

result_dict=dict()
df_results_AR, df_results_RMSNE = pd.DataFrame(), pd.DataFrame()

for path in paths:
    ws, bs, patience, hidden, lr, weight_decay, layer_num, dropout = get_param_value(path)
    result_dict[",".join([bs, patience, hidden, lr, weight_decay, layer_num, dropout])] = dict()
    result_dict[",".join([bs, patience, hidden, lr, weight_decay, layer_num, dropout])][ws] = dict()

    _df = pd.read_csv(f"{all_results_path}/{path}", index_col=0)
    results_AR = _df.loc[AR, ['average']].transpose()
    results_RMSNE = _df.loc[RMSNE, ['average']].transpose()
    result_dict[",".join([bs, patience, hidden, lr, weight_decay, layer_num, dropout])][ws] = [results_AR, results_RMSNE]
    print()
    print(", ".join([bs, patience, hidden, lr, weight_decay, layer_num, dropout]), f', ws={ws}, AR,\t', ', '.join(["{:.2%}".format(v) for v in results_AR.values.flatten()]))
    print(", ".join([bs, patience, hidden, lr, weight_decay, layer_num, dropout]), f', ws={ws}, RMSNE,\t', ', '.join(["{:.2f}".format(v) for v in results_RMSNE.values.flatten()]))