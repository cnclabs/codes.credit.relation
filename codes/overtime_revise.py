import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta


def revise_overtime(data_dir):
    """
    revise those training data with prediction horizon overlapped the testing period
    change to -1
    """
    df_train = pd.read_csv(f'{data_dir}/train_cum.gz')
    df_test = pd.read_csv(f'{data_dir}/test_cum.gz')

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
    return df_train
if __name__ == '__main__':

    kfolds = ["{:02d}".format(i) for i in range(1,14)]
    window_size=['01', '06', '12']

    for ws in window_size:
        for k in kfolds:
            data_dir = f'/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_time/len_{ws}/time_fold_{k}'
            output_dir = f'/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/processed/sample/8_labels_time_no_overtime/len_{ws}/time_fold_{k}'
            df_train = revise_overtime(data_dir)
            df_train.to_csv(f'{output_dir}/train_cum.gz', compression='gzip', index=False)