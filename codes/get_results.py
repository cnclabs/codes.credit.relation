import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--exp_dir', default='/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/experiments', help='where the experiment conduct')
parser.add_argument('--exp_model', default='gru', help='gru, lstm, mlp, ...')
parser.add_argument('--exp_type', default='index', help='index, time')
parser.add_argument('--exp_date', default='mmdd', help='the date')
parser.add_argument('--exp_desc', required=False)
parser.add_argument('--output_dir', default='/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/results', help='output directory')

args = parser.parse_args()

def main(args):
    # recall -> RMSNE
    # cap -> AR
    AR = [f'cap_0{i}' for i in range(1,9)]
    RMSNE = [f'recall_0{i}' for i in range(1,9)]

    # name = args.exp_dir_name # 'cwlin'
    model = args.exp_model # 'gru'
    description = args.exp_desc
    date = args.exp_date #'1125'
    experiment_dir = args.exp_dir

    output_dir = f'{args.output_dir}/{date}-{description}' if description!=None else f'{args.output_dir}/{date}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # cross sectional
    if args.exp_type == 'index':
        path1 = f'{experiment_dir}/{model}01_{args.exp_type}/results_pt.csv'
        path2 = f'{experiment_dir}/{model}06_{args.exp_type}/results_pt.csv'
        path3 = f'{experiment_dir}/{model}12_{args.exp_type}/results_pt.csv'
        
    elif args.exp_type == 'time':
        
        path1 = f'{experiment_dir}/{model}01/results_pt.csv'
        path2 = f'{experiment_dir}/{model}06/results_pt.csv'
        path3 = f'{experiment_dir}/{model}12/results_pt.csv'
    else:
        print('Error')

    print('Path:\n{}\n{}\n{}'.format(path1, path2, path3))
    
    df1 = pd.read_csv(path1, index_col=0)
    df2 = pd.read_csv(path2, index_col=0)
    df3 = pd.read_csv(path3, index_col=0)

    _df1 = df1.loc[AR, ['average']].transpose()
    _df2 = df2.loc[AR, ['average']].transpose()
    _df3 = df3.loc[AR, ['average']].transpose()

    df_results = _df1.append(_df2).append(_df3)

    df_results.index = [f'{model}01', f'{model}06', f'{model}12']
    df_results.columns = [f'AR_0{i}' for i in range(1,9)]
    df_results.to_csv(f'{output_dir}/{model}_{args.exp_type}_AR.csv', index=False, header=None, sep='\t')

    _df1 = df1.loc[RMSNE, ['average']].transpose()
    _df2 = df2.loc[RMSNE, ['average']].transpose()
    _df3 = df3.loc[RMSNE, ['average']].transpose()

    df_results = _df1.append(_df2).append(_df3)

    df_results.index = [f'{model}01', f'{model}06', f'{model}12']
    df_results.columns = [f'RMSNE_0{i}' for i in range(1,9)]
    df_results.to_csv(f'{output_dir}/{model}_{args.exp_type}_RMSNE.csv', index=False, header=None, sep='\t')

if __name__ == '__main__':
    main(args)