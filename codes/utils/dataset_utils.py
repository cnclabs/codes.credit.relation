# utils/dataset_utils.py

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Setup PyTorch Data Loader
class All_Company_Dataset(Dataset):
    def __init__(self, x=None, y=None):
        self._x = x
        self._y = y

    def __getitem__(self, index):
        input_dict = {}
        input_dict['features'] = self._x[index]
        input_dict['labels'] = self._y[index]
            
        return input_dict['features'], input_dict['labels']

    def __len__(self):
        return len(self._x)
    
# Load Data
def load_dataset(is_training, filename, batch_size, feature_size, window_size, company_id_list):
    # Check the compression
    compression = "gzip" if ".gz" in filename else None
    # Get infos, features, and labels (No for_column)
    # Read the data & skip the header
    all_df = pd.read_csv(filename, compression=compression, header=0)    

    # Fill missing values
    features_coverage = 2 + feature_size * window_size
    all_df.iloc[:, :2] = all_df.iloc[:, :2].fillna("") # filled missing values in info columns (date, id) with an empty string
    all_df.iloc[:, 2:features_coverage] = all_df.iloc[:, 2:features_coverage].fillna(0.0) # feature_df
    all_df.iloc[:, features_coverage:] = all_df.iloc[:, features_coverage:].fillna(0) # label_df
    
    # Replace other events as 0: 2 -> 0
    all_df.iloc[:, features_coverage:] = all_df.iloc[:, features_coverage:].replace(2, 0) # label_df
    
    # get all features
    x, y = [], []
    results_dict = dict()

    date_group = all_df.groupby('date')
    for date in all_df.date.sort_values().unique():
        df = date_group.get_group(date)
        df_date_id = df.sort_values(by='id').set_index('id')

        # create rows with all companies fill with 0 if no data exists else fill with original data
        df_all_company_at_t = pd.DataFrame(0, index=company_id_list, columns=df.columns)
        df_all_company_at_t.loc[df_date_id.index, :] = df_date_id # fill original data to df_all_company_at_t if value exists
        df_all_company_at_t['id'] = df_all_company_at_t.index

        # extracts label values from df_all_company_at_t
        label_df = df_all_company_at_t.loc[:, ["y_cum_{:02d}".format(h) for h in range(1, 1+8)]]
        label_df['y_cum_09'] = 1 # every company will default in the infinite future
        label_df.loc[label_df.index.difference(df_date_id.index), :] = -1
        label = np.array(label_df.values, dtype=np.int32)

        df_all_company_at_t.index = range(len(df_all_company_at_t))
        results_dict[date] = df_all_company_at_t.loc[df_all_company_at_t.id.isin(df_date_id.index), :][['date', 'id']]

        # time-lagged observations at time t-delta+1, ... t-1, where delta can be 1,6,12
        feature_window = []
        for rnn_length in range(1, window_size+1):
            # feature
            feature_df = df_all_company_at_t.loc[:, ['x_fea_{:02d}_w_{:02d}'.format(feat_i, rnn_length) for feat_i in range(1, feature_size+1)]]
            feature = np.array(feature_df.values, dtype=np.float32)
            feature_window.append(feature)
        feature_window = np.stack(feature_window, axis=0)

        x.append(feature_window) # 325 * (6, 15786, 14)
        y.append(label)         # 325 * (15786, 9)
        
    x = np.stack(x) # (325, 6, 15786, 14)
    y = np.stack(y) # (325, 15786, 9)

    dataset = All_Company_Dataset(x=x, y=y)
    iterator = DataLoader(dataset, batch_size=batch_size, shuffle=True if is_training else False)
    return iterator, results_dict