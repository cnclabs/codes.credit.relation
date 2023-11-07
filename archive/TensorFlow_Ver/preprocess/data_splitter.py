import os
import click
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from dateutil.relativedelta import relativedelta

# set random seed
SEED = 123
random_state = np.random.RandomState(SEED)

# start date
END_YEAR = '2012-12-31'
WINDOW = 10  # 10 years


def time_splitter(input_df, end_year, window):

    test_end = pd.Timestamp(end_year)
    for i in range(13):
        test_start = test_end - relativedelta(years=1)
        train_end = test_start - relativedelta(days=1)
        train_start = train_end - relativedelta(years=window)
        print("TRAIN: ", train_start, train_end)
        print("TEST: ", test_start, test_end)
        train_idx = input_df.index[
                input_df['date'].between(train_start, train_end)].values
        test_idx = input_df.index[
                input_df['date'].between(test_start, test_end)].values
        test_end = test_start
        yield train_idx, test_idx


def dump_k_fold(input_df, idx_generator, output_types, output_path, split_type):

    for idx, indicies in enumerate(idx_generator):
        fold_path = os.path.join(output_path, split_type + '_fold_{:02d}'.format(idx + 1))

        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

        print("Fold {}: Dump file to csv...".format(idx + 1))
        for i, file_type in enumerate(['train', 'test']):
            file_path = os.path.join(fold_path, file_type)
            df_out = input_df.iloc[indicies[i]]

            for k, v in output_types.items():
                df_out.loc[:, v].to_csv('_'.join([file_path, k]) + '.gz',
                        compression='gzip',
                        index=False)
        print("Fold {}: Finished.".format(idx + 1))


@click.command()
@click.option('--input_file', default='../data/interim/merged.csv')
@click.option('--output_path', default='../data/processed')
@click.option('--mode', default='index')
def main(input_file, output_path, mode):
    """ generate K-fold splits
    """
    # read full dataset
    df = pd.read_csv(input_file)
    df['date'] = pd.to_datetime(df['date'])
    print('original shape: {}'.format(df.shape))

    # Top-level 10-fold split for testing scores
    # sub-level 10-fold split for hyperparameter searching

    k_ids = [c for c in df.columns if 'id' in c or 'date' in c]
    x_fea = [c for c in df.columns if 'x_fea' in c]
    x_cpd = [c for c in df.columns if 'x_cpd' in c]
    y_for = [c for c in df.columns if 'y_for_' in c]
    y_cum = [c for c in df.columns if 'y_cum_' in c]

    cum_cols = k_ids + x_fea + y_cum
    for_cols = k_ids + x_fea + y_for
    fim_cols = k_ids + x_cpd + y_cum
    cum_for_cols = k_ids + x_fea + y_cum + y_for

    #out_types = {'cum': cum_cols, 'for':for_cols, 'fim':fim_cols}
    out_types = {'cum': cum_cols}
    #out_types = {'fim': fim_cols}
    #out_types = {'cum_for': cum_for_cols}

    if mode == 'index':
        # ----------- Split by index ------------ #
        # Consider # of sum cumulative defaults
        # over different forward months' distribution
        # doesn't consider -1 & 2 events
        y = df[y_cum].replace({-1:0, 2:0}).sum(axis=1).values
        dummy = np.zeros(df.shape[0])
        skf = StratifiedKFold(n_splits=13, shuffle=True, random_state=random_state)
        splitter = skf.split(dummy, y)
    else:
        splitter = time_splitter(df, END_YEAR, WINDOW)

    dump_k_fold(df, splitter, out_types, output_path, mode)


if __name__ == "__main__":
    main()
