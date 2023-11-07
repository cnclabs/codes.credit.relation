import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--ROOT", default='/home/cwlin/explainable_credit/data', type=str, help='ROOT')
args = parser.parse_args()

def revise_overtime(data_dir):
    """
    revise those training data with prediction horizon overlapped the testing period
    change to -1
    """
    df_train = pd.read_csv(f'{data_dir}/train_cum.gz')
    df_train_subset = pd.read_csv(f'{data_dir}/train_subset_cum.gz')
    df_valid_subset = pd.read_csv(f'{data_dir}/valid_subset_cum.gz')
    df_test = pd.read_csv(f'{data_dir}/test_cum.gz')

    # train all
    test_date_start = df_test.date.sort_values().unique()[0]
    test_date_start_obj = datetime.strptime(test_date_start, '%Y-%m-%d')
    date_group = df_train.groupby('date')

    for train_date in df_train.date.sort_values().unique():
        current_date_obj = datetime.strptime(train_date, '%Y-%m-%d')
        col = []
        for prediction_horizon, y_cum in zip([1,3,6,12,24,36,48,60], list(range(1,9))):
            pred_date_obj = current_date_obj + relativedelta(months=prediction_horizon)
            if test_date_start_obj<=pred_date_obj:
                col.append(y_cum)
        if len(col)>0:
            col = [f'y_cum_0{c}' for c in col]
            idx = date_group.get_group(train_date).index
            df_train.loc[idx, col] = -1
    
    # train_subset
    valid_date_start = df_valid_subset.date.sort_values().unique()[0]
    valid_date_start_obj = datetime.strptime(valid_date_start, '%Y-%m-%d')
    date_group = df_train_subset.groupby('date')

    for train_date in df_train_subset.date.sort_values().unique():
        current_date_obj = datetime.strptime(train_date, '%Y-%m-%d')
        col = []
        for prediction_horizon, y_cum in zip([1,3,6,12,24,36,48,60], list(range(1,9))):
            pred_date_obj = current_date_obj + relativedelta(months=prediction_horizon)
            if valid_date_start_obj<=pred_date_obj:
                col.append(y_cum)
        if len(col)>0:
            col = [f'y_cum_0{c}' for c in col]
            idx = date_group.get_group(train_date).index
            df_train_subset.loc[idx, col] = -1
    
    
    # valid_subset
    test_date_start = df_test.date.sort_values().unique()[0]
    test_date_start_obj = datetime.strptime(test_date_start, '%Y-%m-%d')
    date_group = df_valid_subset.groupby('date')

    for train_date in df_valid_subset.date.sort_values().unique():
        current_date_obj = datetime.strptime(train_date, '%Y-%m-%d')
        col = []
        for prediction_horizon, y_cum in zip([1,3,6,12,24,36,48,60], list(range(1,9))):
            pred_date_obj = current_date_obj + relativedelta(months=prediction_horizon)
            if test_date_start_obj<=pred_date_obj:
                col.append(y_cum)
        if len(col)>0:
            col = [f'y_cum_0{c}' for c in col]
            idx = date_group.get_group(train_date).index
            df_valid_subset.loc[idx, col] = -1

    return df_train, df_train_subset, df_valid_subset, df_test
if __name__ == '__main__':

    kfolds = ["{:02d}".format(i) for i in range(1,14)]
    # window_size=['01', '06', '12']

    # for ws in window_size:
    #     for k in kfolds:
    #         data_dir = f'{args.ROOT}/8_labels_time/len_{ws}/time_fold_{k}'
    #         output_dir = f'{args.ROOT}/8_labels_time_no_overtime/len_{ws}/time_fold_{k}'
    #         if not os.path.exists(output_dir):
    #             os.makedirs(output_dir)
    #         df_train, df_test = revise_overtime(data_dir)
    #         df_train.to_csv(f'{output_dir}/train_cum.gz', compression='gzip', index=False)
    #         df_test.to_csv(f'{output_dir}/test_cum.gz', compression='gzip', index=False)

    # for ws in window_size:
    for k in kfolds:
        data_dir = f'{args.ROOT}/expand_01_cum_for/time_fold_{k}'
        output_dir = f'{args.ROOT}/expand_01_cum_for_no_overtime/time_fold_{k}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_train, df_train_subset, df_valid_subset, df_test = revise_overtime(data_dir)
        df_train.to_csv(f'{output_dir}/train_cum.gz', compression='gzip', index=False)
        df_train_subset.to_csv(f'{output_dir}/train_subset_cum.gz', compression='gzip', index=False)
        df_valid_subset.to_csv(f'{output_dir}/valid_subset_cum.gz', compression='gzip', index=False)
        df_test.to_csv(f'{output_dir}/test_cum.gz', compression='gzip', index=False)