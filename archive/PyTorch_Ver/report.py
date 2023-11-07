import click
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from subprocess import check_call
from main import *
import ar_cap


def clean_data(file_path):
    #FIXME: manually handle abnormal syntax in file
    input_str = file_path.split('/')
    filename = input_str[-1]
    fixed_file = "/".join(input_str[:-1] + ['fixed_' + filename])
    cmd = 'cat {} \
            | sed "s/b\'//g" \
            | sed "s/\'//g" > {}'.format(file_path, fixed_file)
    check_call(cmd, shell=True)
    return fixed_file


def agg_rmse(y_pred, y_true, norm_factor=1, factor=1):
    """
    args:
        y_pred: (np.array)
        y_true: (np.array)
        norm_factor: (np.array or constant)
    """
    # only keep items not inf
    # because y_true could be zeros in some months
    scores = np.setdiff1d(
            (y_pred - y_true)**2 / (norm_factor)**2, 
             np.inf) * (factor ** 2)
    return np.sqrt(np.mean(scores))


@click.command()
@click.option('--pred_file', default=None, help='prediction file')
@click.option('--label_file', default=None, help='label file')
def main(pred_file, label_file):

    file_type = label_file.split('/')[-1].split('_')[0]
    print(file_type)

    if '.gz' not in pred_file:
        fixed_file = clean_data(pred_file)
    else:
        fixed_file = pred_file

    # file I/O
    df_cpd = pd.read_csv(fixed_file, index_col=False, engine='python')
    df_cum = pd.read_csv(label_file, index_col=False, engine='python')
    print(df_cpd)

    assert df_cpd.shape[0] == df_cum.shape[0], "Number of rows is different,\
            plz check consistency of pred_file and label_file"

    df_cpd['date'] = pd.to_datetime(df_cpd['date'])
    df_cum['date'] = pd.to_datetime(df_cum['date'])
    df_cpd['id'] = df_cpd['id'].astype('int32')
    df_cum['id'] = df_cum['id'].astype('int32')
    key_cols = ['id', 'date']

    preds = [c for c in df_cpd.columns if 'p_cum' in c or 'x_cpd' in c]
    label = [c for c in df_cum.columns if 'y_cum' in c]
    df_cpd = df_cpd.loc[:, key_cols + preds]
    df_cum = df_cum.loc[:, key_cols + label]
    print('Start to merge pred/label files')
    df = pd.merge(left=df_cpd, right=df_cum, on=['id', 'date'])
    print('Merge done.')
    assert df.shape[0] != 0, "No matched date/ID,\
            plz check consistency of pred_file and label_file"

    # FIXME: Other exit -> alive
    df = df.replace({2:0})

    # FIXME: Hard coded masking
    mask_list = [pd.Timestamp(2017,12,1), pd.Timestamp(2017,10,1),
                 pd.Timestamp(2017,7,1), pd.Timestamp(2017,1,1),
                 pd.Timestamp(2016,1,1), pd.Timestamp(2015,1,1),
                 pd.Timestamp(2014,1,1), pd.Timestamp(2013,1,1)]

    metrics = {}
    for i in range(len(label)):
        ## ----------- Normal Metrics ----------- #
        # FIXME: doesn't count any instance with -1 label
        true_col = label[i]
        pred_col = preds[i]
        mask_date = (df['date'] < mask_list[i])
        mask_invalid = (df[true_col] != -1)
        final_mask = mask_date & mask_invalid

        _df = df.loc[final_mask, ['date', pred_col, true_col]]

        y_pred = _df[pred_col].values
        y_true = _df[true_col].values
        metrics["auc_{:02d}".format(i + 1)] = roc_auc_score(y_true, y_pred)
        metrics["cap_{:02d}".format(i + 1)] = ar_cap.cap_ar_score(y_true, y_pred)

        ## ----------- Aggregated Metrics ----------- #
        _df = _df.groupby('date')

        size = _df.size().values
        _df = _df.sum().reset_index()
        # Num of exist companies of each month
        y_pred = _df[pred_col].values
        y_true = _df[true_col].values

        metrics["rmse_{:02d}".format(i + 1)]   = agg_rmse(y_pred, y_true)
        metrics["rate(%)_{:02d}".format(i + 1)]   = agg_rmse(y_pred, y_true, size, factor=100)
        metrics["recall_{:02d}".format(i + 1)] = agg_rmse(y_pred, y_true, y_true)

    #metrics_string = "; ".join("{}: {:.3f}".format(k, v) for k, v in metrics.items())
    #print(metrics_string)
    input_str = pred_file.split('/')
    filename = input_str[-1]
    # Original: metrics_pred_best_weights.json -> metrics_pred_best_weights_new.json
    json_file = "/".join(input_str[:-1] +
            ['{}_metrics_pred_best_weights.json'.format(file_type)])
    save_dict_to_json(metrics, json_file)


if __name__ == "__main__":
    main()
