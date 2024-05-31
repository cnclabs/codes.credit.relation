import time
from re import X
from model import *
from utils import *
import torch
import torch.nn.functional as F
from torch import optim
import argparse
import logging
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import trange
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.utils.data as data
import pickle
from torch.optim.lr_scheduler import ExponentialLR
import random

parser = argparse.ArgumentParser()
# experiments settings
parser.add_argument('--model_dir', default='./experiments/gru12_index')
parser.add_argument('--model_name', default='NeuDP')
parser.add_argument('--data_dir', default='/home/cwlin/explainable_credit/data', help="Directory containing the dataset")
parser.add_argument('--device', default='1', help="CUDA_DEVICE")
parser.add_argument('--all_company_ids_path', default='/home/cwlin/explainable_credit/data/all_company_ids.csv', help="path to list of all company ids")
parser.add_argument('--feature_size', default=14, type=int)
parser.add_argument('--window_size', default=12, type=int)
parser.add_argument('--cum_labels', default=8, type=int)

# training settings
# parser.add_argument('--search_hyperparameter', default=False, action='store_true')
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--patience', default=20, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float, help='1e-3, 1e-4, 1e-5')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='1e-4, 1e-5, 1e-6, l2 regularization')
parser.add_argument('--gamma', default=0.9, type=float, help='exponential learning rate scheduler, lr = lr_0 * gamma ^ epoch')

# model architecture settings
parser.add_argument('--lstm_num_units', type=int, required=True)
# parser.add_argument('--dropout_rate', type=float, required=True)

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

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, logits, labels):
        pad_event = torch.tensor(-1)
        not_pad = torch.ne(labels, pad_event)
        masks = not_pad.type(torch.float32)

        losses = (-1) * ( labels * torch.log(logits) + (1 - labels) * torch.log(1 - logits) )
        # loss = torch.mean(losses * masks)
        loss = torch.sum(losses * masks) / torch.sum(masks)
        return loss

# Evaluation metrics
def safe_auc(y_true, y_score):
    y_true, y_score = torch.tensor(y_true), torch.tensor(y_score)
    mask = y_true.ne(-1)
    y_true = torch.masked_select(y_true, mask)
    y_score = torch.masked_select(y_score, mask)
    if len(np.unique(y_true))==0:
        return 0.0
    elif len(np.unique(y_true))==1:
        return accuracy_score(y_true, np.rint(y_score))
    else:
        return roc_auc_score(y_true, np.rint(y_score))

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

def train(model, cum_labels, train_inputs, device):
    model.train()
    
    metrics = {
        'batch_aucs':torch.empty((0, cum_labels), dtype=torch.float64).to(device),
        'batch_loss':torch.empty((0), dtype=torch.float64).to(device),
        'aucs': torch.empty((0, cum_labels), dtype=torch.float64).to(device),
        'loss': torch.empty((0), dtype=torch.float64).to(device)
    }

    t = trange(len(train_inputs))
    for i, (features, labels) in zip(t, train_inputs):
        # Zero gradients for every batch
        optimizer.zero_grad()

        batch_size = features.size(0)
        for batch in range(batch_size):
            # Make predictions for this batch
            inputs = features[batch].to(device)
            targets = labels[batch].to(device)
            predict = model(inputs)
            
            # Compute the loss and its gradients
            loss = criterion(predict, targets)
            loss.backward()

            batch_aucs = torch.tensor([[safe_auc(y_true=targets[:, i].cpu().detach().numpy(), 
                            y_score=predict[:, i].cpu().detach().numpy())
                            for i in range(cum_labels)]]).to(device)
            batch_loss = torch.tensor([loss]).to(device) # 1

            metrics['batch_aucs'] = torch.cat((metrics['batch_aucs'], batch_aucs), 0)
            metrics['batch_loss'] = torch.cat((metrics['batch_loss'], batch_loss), 0)
        metrics['batch_aucs'] = torch.mean(metrics['batch_aucs'], 0, keepdim=True) # [4, 8] -> [1, 8]
        metrics['batch_loss'] = torch.mean(metrics['batch_loss'], 0, keepdim=True) # [4] -> [1]
        loss = metrics['batch_loss'].item()
        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss))

        # Adjust learning weights
        optimizer.step()
        
        metrics['aucs'] = torch.cat((metrics['aucs'], metrics['batch_aucs']), 0) # [1, 8]
        metrics['loss'] = torch.cat((metrics['loss'], metrics['batch_loss']), 0) # [1]

    metrics['aucs'] = torch.mean(metrics['aucs'], 0)
    metrics['loss'] = torch.mean(metrics['loss'])

    k_aucs = ['auc_{}'.format(i+1) for i in range(cum_labels)]
    v_aucs = metrics['aucs'].tolist()
    metrics_val = dict(zip(k_aucs, v_aucs))
    metrics_val['auc'] = torch.mean(metrics['aucs']).item()
    metrics_val['loss'] = metrics['loss'].item()
    metrics_string = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_val


def evaluate(model, cum_labels, eval_inputs, device):
    # Flag as evaluation
    model.eval()

    metrics = {
        'aucs': torch.empty((0, int(cum_labels)), dtype=torch.float64).to(device),
        'batch_aucs': torch.empty((0, int(cum_labels)), dtype=torch.float64).to(device),
        'loss': torch.empty((0), dtype=torch.float64).to(device),
        'batch_loss': torch.empty((0), dtype=torch.float64).to(device)
    }


    t = trange(len(eval_inputs))
    for i, (features, labels) in zip(t, eval_inputs):
        batch_size = features.size(0)
        
        
        for batch in range(batch_size):
            inputs = features[batch].to(device)
            targets = labels[batch].to(device)

            predict = model(inputs)

            # Compute the loss and its gradients
            loss = criterion(predict, targets)
            loss = loss.item()

            batch_aucs = torch.tensor([[safe_auc(y_true=targets[:, i].cpu().detach().numpy(), 
                            y_score=predict[:, i].cpu().detach().numpy())
                            for i in range(cum_labels)]]).to(device)
            batch_loss = torch.tensor([loss]).to(device)
            metrics['batch_aucs'] = torch.cat((metrics['batch_aucs'], batch_aucs), 0)
            metrics['batch_loss'] = torch.cat((metrics['batch_loss'], batch_loss), 0)
        metrics['batch_aucs'] = torch.mean(metrics['batch_aucs'], 0, keepdim=True) # [4, 8] -> [1, 8]
        metrics['batch_loss'] = torch.mean(metrics['batch_loss'], 0, keepdim=True) # [4] -> [1]
    
        metrics['aucs'] = torch.cat((metrics['aucs'], metrics['batch_aucs']), 0) # [1, 8]
        metrics['loss'] = torch.cat((metrics['loss'], metrics['batch_loss']), 0) # [1]
    
    metrics['aucs'] = torch.mean(metrics['aucs'], 0)
    metrics['loss'] = torch.mean(metrics['loss'])

    k_aucs = ['auc_{}'.format(i+1) for i in range(cum_labels)]
    v_aucs = metrics['aucs'].tolist()
    metrics_val = dict(zip(k_aucs, v_aucs))
    metrics_val['auc'] = torch.mean(metrics['aucs']).item()
    metrics_val['loss'] = metrics['loss'].item()
    metrics_string = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Eval metrics: " + metrics_string)
    
    # For calculating metrics
    return metrics_val
    
if __name__ == '__main__':
    args = parser.parse_args()
    save_args_to_json(args, args.model_dir)

    model_dir = args.model_dir
    data_dir = args.data_dir
    all_company_ids_path = args.all_company_ids_path
    device = args.device
    cum_labels = args.cum_labels

    # Set the random seed
    set_seed(230)

    # setting for individual device allocating
    device = torch.device(f"cuda:{device}" if device != "cpu" and torch.cuda.is_available() else "cpu")

    # Set the logger
    set_logger(os.path.join(model_dir, 'train.log'))

    # Args
    logging.info(f"{args}")

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    logging.info("Loading all train dataset...")
    train_data = os.path.join(data_dir, 'train_cum.gz')
    all_company_ids = pd.read_csv(f'{all_company_ids_path}', index_col=0).id.sort_values().unique()    
    
    start_t = time.time()
    train_all_inputs, _ = load_dataset(True, train_data, args.batch_size, args.feature_size, args.window_size, all_company_ids)
    end_t = time.time()
    logging.info("load all train dataset: {}s".format(end_t - start_t))

    model = NeuDP(args.feature_size, args.window_size, cum_labels+1, args.lstm_num_units, num_company=len(all_company_ids), device=device)
    model.to(device)
    model.to(torch.float)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    criterion = CustomCrossEntropyLoss()

    # Start training all
    logging.info("Start training all...")
    final_num_epoch = args.num_epochs
    for epoch in range(final_num_epoch):
        # Train & Evaluate
        logging.info("Epoch {}/{}:".format(epoch + 1, final_num_epoch))
        train_ret_metric = train(model, args.cum_labels, train_all_inputs, device)
        scheduler.step()

    # Save the final model
    if not os.path.exists(f'{args.model_dir}/last_weights'):
        os.makedirs(f'{args.model_dir}/last_weights')
    final_save_path = os.path.join(model_dir, 'last_weights', f'{args.model_name}_checkpoint.pt')
    torch.save(model.state_dict(), final_save_path)
