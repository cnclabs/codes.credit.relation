









import os
import click

from functools import reduce

import pandas as pd
import numpy as np


def parse_csv(path):
    """parse and merge clean csv into merged dataframe.
    :path: (string)
    :returns: (DataFrame)
    """
    label_cum = os.path.join(path, 'label_cum.csv')
    label_for = os.path.join(path, 'label_for.csv')
    data_fea = os.path.join(path, 'data_fea.csv')
    data_cpd = os.path.join(path, 'data_cpd.csv')
    names_gen = ['id', 'year', 'month']
    names_cum = ['y_cum_{:02d}'.format(i + 1) for i in range(8)]
    names_for = ['y_for_{:02d}'.format(i + 1) for i in range(60)]
    names_fea = ['x_fea_{:02d}'.format(i + 1) for i in range(14)]
    names_cpd = ['x_cpd_{:02d}'.format(i + 1) for i in range(6)]
    def df_parse(_file, names):
        print("Parsing...{}".format(_file))
        return pd.read_csv(_file, header=None,
                names=names,
                parse_dates={'date':[1,2]})
    df_cum = df_parse(label_cum, names=names_gen + names_cum)
    df_for = df_parse(label_for, names=names_gen + names_for)
    df_fea = df_parse(data_fea, names=names_gen + names_fea)
    df_cpd = df_parse(data_cpd, names=names_gen + names_cpd)
    dfs = [df_for, df_cum, df_fea, df_cpd]
    print('Merge DataFrames...')
    df = reduce(lambda left, right:
            pd.merge(left, right, on=['id', 'date']), dfs)

    return df


def sliding_window(seq, window_size, step_size=1):
    #for i in range(len(seq), 0, -step_size):
    for i in range(0, len(seq), step_size):
        yield seq[max(i - window_size + 1, 0): i+1]


def gen_seq(infos, features, indicies, window_size, step_size):

    for i, _ in enumerate(indicies):
        start = indicies[i]
        if i >= len(indicies) - 1:
            stop = None
        else:
            stop = indicies[i + 1]
        for idx, seq in enumerate(sliding_window(features[start:stop],
                window_size, step_size)):
            yield (infos[(start + idx)], seq)


@click.command()
@click.option('--path', default='../data/interim/')
@click.option('--window', default=12)
@click.option('--step', default=1)
def main(path, window, step):
    merged_df_path = os.path.join(path, 'merged.csv')

    if not os.path.isfile(merged_df_path):
        df = parse_csv(path)
        df = df.reindex(sorted(df.columns), axis=1)
        df.to_csv(merged_df_path, index=False)
    else:
        df = pd.read_csv(merged_df_path)

    fea_cols = [col for col in df.columns if 'x_fea' in col]
    info_cols = [col for col in df.columns if 'x_fea' not in col]
    # features: x_fea_1~x_fea_14
    features = df.loc[:, fea_cols].values
    # infos: date/id/cpd_1 ~ cpd_6/cum_1 ~ cum_6/for_1 ~ for_60
    infos = df.loc[:, info_cols].values
    df['date'] = pd.to_datetime(df['date'])
    company_id = df['id'].values
    date = df['date'].values

    del df

    # sort by company_id then by date
    sort_idx = np.lexsort((date, company_id))
    features = features[sort_idx]
    infos = infos[sort_idx]

    # get indicies by company_ids
    company_ids, indices, counts = np.unique(infos[:, 1],
            return_index=True,
            return_counts=True)

    # each record will have a subsequence with window
    num_subseqs = int(np.ceil(counts / float(step)).sum())

    # padding 0 for feature values
    seq_feas = np.zeros((num_subseqs, window, features.shape[1]), dtype=np.float32)
    seq_info = np.empty((num_subseqs, infos.shape[1]), dtype=np.object)

    for idx, (info, seq) in enumerate(gen_seq(infos,
                                             features,
                                             indices,
                                             window,
                                             step)):
        #seq_feas[idx][-len(seq):] = seq
        seq_feas[idx][:len(seq)] = seq
        seq_info[idx] = info

    seq_feas = seq_feas.reshape(-1, window*len(fea_cols))
    data = np.concatenate((seq_info, seq_feas), axis=1)
    fea_cols = [col + '_w_{:02d}'.format(w + 1) for w in range(window) for col in fea_cols]
    df = pd.DataFrame(data, columns=info_cols + fea_cols)
    df.to_csv(path + 'seq_w_{:02d}_s_{:02d}.csv'.format(window, step), index=False)


if __name__ == "__main__":
    main()
