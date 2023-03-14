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

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
parser.add_argument('--label_type', default='cum', help="Label types, cum only or cum with forward label")
parser.add_argument('--restore_dir', default='None',
                    help="Optional, directory containing weights to reload before training")
parser.add_argument('--device', default='2', help="CUDA_DEVICE")
parser.add_argument('--all_company_ids_path', default='data/all_company_ids.csv', help="path to list of all company ids")

parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--learning_rate', default=1e-4, type=float)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--weight_decay', default=1e-5, type=float)
parser.add_argument('--patience', default=20, type=int)
parser.add_argument('--lstm_num_units', default=64, type=int)
parser.add_argument('--shared_param', default=False, action='store_true')

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
def load_dataset(is_training, filename, params, args, all_company, device, is_training_all=False):
    # Check the compression
    compression = "gzip" if ".gz" in filename else None
    
    # Get infos, features, and labels (No for_column)
    # Read the data & skip the header
    all_df = pd.read_csv(filename, compression=compression, header=0)
    
    # Fill missing values
    features_coverage = 2 + params.feature_size * params.window_size
    all_df.iloc[:, :2] = all_df.iloc[:, :2].fillna("") # info_df
    all_df.iloc[:, 2:features_coverage] = all_df.iloc[:, 2:features_coverage].fillna(0.0) # feature_df
    all_df.iloc[:, features_coverage:] = all_df.iloc[:, features_coverage:].fillna(0) # label_df
    
    # Replace other events as 0: 2 -> 0
    all_df.iloc[:, features_coverage:] = all_df.iloc[:, features_coverage:].replace(2, 0)
    
    date_group = all_df.groupby('date')

    # get all features
    x, y = [], []
    results_dict = dict()

    for date in all_df.date.sort_values().unique():
        df = date_group.get_group(date)
        # create rows with all companies fill with 0 if no data exists else fill with original data
        df_date_id = df.sort_values(by='id').set_index('id')

        df_all_company_at_t = pd.DataFrame(0, index=all_company, columns=df.columns)
        df_all_company_at_t.loc[df_date_id.index, :] = df_date_id
        df_all_company_at_t['id'] = df_all_company_at_t.index

        label_df = df_all_company_at_t.loc[:, ["y_cum_{:02d}".format(h) for h in range(1,1+8)]]
        label_df['y_cum_09'] = 1
        label_df.loc[label_df.index.difference(df_date_id.index), :] = -1
        label = torch.tensor(label_df.values, dtype = torch.int).to(device)

        df_all_company_at_t.index = range(len(df_all_company_at_t))
        results_dict[date] = df_all_company_at_t.loc[df_all_company_at_t.id.isin(df_date_id.index), :][['date', 'id']]

        
        # time-lagged observations at time t-delta+1, ... t-1, where delta can be 1,6,12
        feature_window, label_window = [], []
        for rnn_length in range(1, params.window_size+1):
            # feature
            feature_df = df_all_company_at_t.loc[:, ['x_fea_{:02d}_w_{:02d}'.format(feat_i, rnn_length) for feat_i in range(1, params.feature_size+1)]]
            feature = torch.tensor(feature_df.values, dtype = torch.float32).to(device)
            feature_window.append(feature)

        feature_window = torch.stack(feature_window, dim=0)

        x.append(feature_window) # 325 * (6, 15786, 14)
        y.append(label)         # 325 * (15786, 9)

    x = torch.stack(x) # (325, 6, 15786, 14)
    y = torch.stack(y) # (325, 6, 15786, 9)
    dataset = All_Company_Dataset(x=x, y=y)

    if is_training:
        if is_training_all:
            train_iterator = DataLoader(dataset, batch_size=args.batch_size, shuffle=True if is_training else False)
            return train_iterator, results_dict
        else:
            train_set, valid_set = data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)], generator=torch.Generator().manual_seed(230))
            train_iterator = DataLoader(train_set, batch_size=args.batch_size, shuffle=True if is_training else False)
            valid_iterator = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True if is_training else False)
            return train_iterator, valid_iterator, results_dict
    else:
        iterator = DataLoader(dataset, batch_size=args.batch_size, shuffle=True if is_training else False)
        return iterator, results_dict
    # iterator = DataLoader(dataset, batch_size=1, shuffle=True if is_training else False)
    # return iterator, results_dict

def train(model, params, train_inputs, device, relation_static = None):
    model.train()
    
    metrics = {
        'batch_aucs':torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device),
        'batch_loss':torch.empty((0), dtype=torch.float64).to(device),
        'aucs': torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device),
        'loss': torch.empty((0), dtype=torch.float64).to(device)
    }
    # seq_len = len(x_train)
    # train_seq = list(range(seq_len))[rnn_length:]
    # random.shuffle(train_seq)
    # total_loss = 0
    # total_loss_count = 0
    # batch_train = 1

    t = trange(len(train_inputs))
    for i, (features, labels) in zip(t, train_inputs):
        # Zero gradients for every batch
        optimizer.zero_grad()

        batch_size = features.size(0)
        for batch in range(batch_size):
            # Make predictions for this batch
            predict = model(features[batch])
            
            # Compute the loss and its gradients
            loss = criterion(predict, labels[batch])
            loss.backward()

            batch_aucs = torch.tensor([[safe_auc(y_true=labels[batch][:, i].cpu().detach().numpy(), 
                            y_score=predict[:, i].cpu().detach().numpy())
                            for i in range(params.cum_labels)]]).to(device)
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
        # loss = loss.item()
        
        metrics['aucs'] = torch.cat((metrics['aucs'], metrics['batch_aucs']), 0) # [1, 8]
        metrics['loss'] = torch.cat((metrics['loss'], metrics['batch_loss']), 0) # [1]

    metrics['aucs'] = torch.mean(metrics['aucs'], 0)
    metrics['loss'] = torch.mean(metrics['loss'])

    k_aucs = ['auc_{}'.format(i+1) for i in range(params.cum_labels)]
    v_aucs = metrics['aucs'].tolist()
    metrics_val = dict(zip(k_aucs, v_aucs))
    metrics_val['auc'] = torch.mean(metrics['aucs']).item()
    metrics_val['loss'] = metrics['loss'].item()
    metrics_string = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)
    return metrics_val


def evaluate(model, params, eval_inputs, device, relation_static = None):
    # Flag as evaluation
    model.eval()

    metrics = {
        'aucs': torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device),
        'batch_aucs': torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device),
        'loss': torch.empty((0), dtype=torch.float64).to(device),
        'batch_loss': torch.empty((0), dtype=torch.float64).to(device)
    }


    t = trange(len(eval_inputs))
    for i, (features, labels) in zip(t, eval_inputs):
        batch_size = features.size(0)
        for batch in range(batch_size):

            predict = model(features[batch])

            # Compute the loss and its gradients
            loss = criterion(predict, labels)
            loss = loss.item()

            batch_aucs = torch.tensor([[safe_auc(y_true=labels[batch][:, i].cpu().detach().numpy(), 
                            y_score=predict[:, i].cpu().detach().numpy())
                            for i in range(params.cum_labels)]]).to(device)
            batch_loss = torch.tensor([loss]).to(device)
            metrics['batch_aucs'] = torch.cat((metrics['batch_aucs'], batch_aucs), 0)
            metrics['batch_loss'] = torch.cat((metrics['batch_loss'], batch_loss), 0)
        metrics['batch_aucs'] = torch.mean(metrics['batch_aucs'], 0, keepdim=True) # [4, 8] -> [1, 8]
        metrics['batch_loss'] = torch.mean(metrics['batch_loss'], 0, keepdim=True) # [4] -> [1]
    
        metrics['aucs'] = torch.cat((metrics['aucs'], metrics['batch_aucs']), 0) # [1, 8]
        metrics['loss'] = torch.cat((metrics['loss'], metrics['batch_loss']), 0) # [1]
    
    metrics['aucs'] = torch.mean(metrics['aucs'], 0)
    metrics['loss'] = torch.mean(metrics['loss'])

    k_aucs = ['auc_{}'.format(i+1) for i in range(params.cum_labels)]
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

    model_dir = args.model_dir # '/tmp2/cwlin/explainable_credit/explainable_credit/experiments/gru06_index/index_fold_06'
    data_dir = args.data_dir # '/tmp2/cwlin/explainable_credit/data/8_labels_index/len_06/index_fold_06'
    all_company_ids_path = args.all_company_ids_path # '/tmp2/cwlin/explainable_credit/data/all_company_ids.csv'
    label_type = args.label_type # 'cum'
    restore_dir = args.restore_dir # 'None'
    device = args.device # '0'
    print('shared_param: {}'.format(args.shared_param))

    # Set the random seed
    set_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    # args = parser.parse_args()

    # setting for individual device allocating
    # os.environ['CUDA_VISIBLE_DEVICES'] = '' if device == 'cpu' else device
    # DEVICE = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    DEVICE = torch.device(f"cuda:{device}")

    # Read the json file with parameters
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")

    # Specify the names of the data files
    train_data = "train_{}.gz".format(params.label_type)
    test_data  = "test_{}.gz".format(params.label_type)

    # Paths to data
    train_data = os.path.join(data_dir, train_data)
    eval_data  = os.path.join(data_dir, test_data)

    # Skip header (Get the sizes)
    # params.train_size = int(os.popen('zcat ' + train_data + '|wc -l').read()) - 1
    # params.eval_size = int(os.popen('zcat ' + eval_data  + '|wc -l').read()) - 1
    # logging.info('Train size: {}, Eval size: {}'.format(params.train_size, params.eval_size))

    # Load dataset
    logging.info("Loading dataset...")

    all_company = pd.read_csv(f'{all_company_ids_path}', index_col=0).id.sort_values().unique()
    train_inputs, valid_inputs, _ = load_dataset(True, train_data, params, args, all_company, DEVICE)
    train_all_inputs, _ = load_dataset(True, train_data, params, args, all_company, DEVICE, is_training_all=True)
    # train_inputs, _ = load_dataset(True, train_data, params, all_company, DEVICE)
    # eval_inputs, _ = load_dataset(False, eval_data, params, all_company, DEVICE)
    # logging.info('Check train: {}, Check eval: {}'.format(len(train_inputs), len(eval_inputs)))
    # logging.info("- done.")


    # hyper-parameters
    NUM_STOCK = len(all_company) # 15786

    # MAX_EPOCH =  300
    # infer = 1 # if infer relation
    # hidn_rnn = 64 # rnn hidden nodes
    # heads_att = 6 # attention heads
    # hidn_att=60 # attention hidden nodes
    # lr = 5e-4
    # rnn_length = x_train[0].size(0) # rnn length 6

    relation="None"
    if relation != "None":
        static = 1
        pass
    else:
        static = 0
        relation_static = None

    # initialize
    best_model_file = 0
    epoch = 0
    wait_epoch = 0
    eval_epoch_best = 0

    # use_relation = False

    # if use_relation:
    #     model = AD_GAT(params=params, num_stock=NUM_STOCK,
    #                         d_hidden = feature_size, hidn_rnn = hidn_rnn, heads_att = heads_att,
    #                         hidn_att= hidn_att,
    #                         infer = infer, relation_static = static)
    # else:
    model = AD_GAT_without_relational(args=args, params=params, num_stock=NUM_STOCK, shared_param=args.shared_param) # 15786, 14
    model.cuda(device=DEVICE)
    model.to(torch.float)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


    # criterion = torch.nn.NLLLoss()
    criterion = CustomCrossEntropyLoss()

    # Start training
    logging.info("Start training...")

    # write loss
    # date = '1126'
    # loss_dir = f'/tmp2/cwlin/explainable_credit/test_results/loss/{date}'
    # train_f = open(f'{loss_dir}/train_loss.epochs={args.num_epochs}', 'a')
    # valid_f = open(f'{loss_dir}/valid_loss.epochs={args.num_epochs}', 'a')
    # train_f.write(f'{model_dir}\n')
    # valid_f.write(f'{model_dir}\n')

    begin_at_epoch = 0
    best_eval_acc = 0.0
    min_valid_loss = float("inf")
    # min_eval_loss = float("inf")
    patience = 0
    for epoch in range(begin_at_epoch, begin_at_epoch + args.num_epochs): 
        # Train & Evaluate
        logging.info("Epoch {}/{}:".format(epoch + 1, begin_at_epoch + args.num_epochs))
        train_ret_metric = train(model, params, train_inputs, DEVICE, relation_static = relation_static)
        valid_ret_metric = evaluate(model, params, valid_inputs, DEVICE, relation_static = relation_static)
        # ret_metrics = evaluate(model, params, eval_inputs, DEVICE, relation_static = relation_static)
        
        # If best_eval, best_save_path
        # eval_acc = ret_metrics['auc']
        valid_acc = valid_ret_metric['auc']
        if valid_acc >= best_eval_acc:
            # Store new best accuracy
            best_eval_acc = valid_acc
            # Save best eval metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
            save_dict_to_json(valid_ret_metric, best_json_path)

        # Save latest eval metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
        save_dict_to_json(valid_ret_metric, last_json_path)
        
        # loss: patience
        train_loss = train_ret_metric['loss']
        valid_loss = valid_ret_metric['loss']
        # train_f.write(f'Epoch: {epoch}\t{train_loss}\n')
        # valid_f.write(f'Epoch: {epoch}\t{valid_loss}\n')

        if valid_loss < min_valid_loss :
            # print(min_valid_loss, ' ---> ', valid_loss)
            min_valid_loss = valid_loss
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                print(f'Early stopping at epoch: {epoch}. Patience={args.patience}')
                final_num_epoch = epoch
                break

    # 2nd initialize model and train again
    model = AD_GAT_without_relational(args=args, params=params, num_stock=NUM_STOCK, shared_param=args.shared_param) # 15786, 14
    model.cuda(device=DEVICE)
    model.to(torch.float)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # criterion = torch.nn.NLLLoss()
    criterion = CustomCrossEntropyLoss()

    # Start training
    logging.info("Start training all...")
    for epoch in range(final_num_epoch):
        # Train & Evaluate
        logging.info("Epoch {}/{}:".format(epoch + 1, final_num_epoch))
        train_ret_metric = train(model, params, train_all_inputs, DEVICE, relation_static = relation_static)
        # valid_ret_metric = evaluate(model, params, valid_inputs, DEVICE, relation_static = relation_static)
        # ret_metrics = evaluate(model, params, eval_inputs, DEVICE, relation_static = relation_static)
        
        # If best_eval, best_save_path
        # eval_acc = ret_metrics['auc']
        # valid_acc = valid_ret_metric['auc']
        # if valid_acc >= best_eval_acc:
        #     # Store new best accuracy
        #     best_eval_acc = valid_acc
        #     # Save best eval metrics in a json file in the model directory
        #     best_json_path = os.path.join(model_dir, "metrics_eval_best_weights.json")
        #     save_dict_to_json(valid_ret_metric, best_json_path)

        # # Save latest eval metrics in a json file in the model directory
        # last_json_path = os.path.join(model_dir, "metrics_eval_last_weights.json")
        # save_dict_to_json(valid_ret_metric, last_json_path)

    # train_f.close()
    # valid_f.close()
    # Save the final model
    final_save_path = os.path.join(model_dir, 'last_weights', 'checkpoint')
    torch.save(model, final_save_path)

