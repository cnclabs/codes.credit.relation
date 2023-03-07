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
import pickle
from torch.utils.data.sampler import SubsetRandomSampler
import warnings
import time

# import EarlyStopping
from pytorchtools import EarlyStopping

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
parser.add_argument('--label_type', default='cum', help="Label types, cum only or cum with forward label", required=False)
parser.add_argument('--restore_dir', default='None',
                    help="Optional, directory containing weights to reload before training", required=False)
parser.add_argument('--device', default='0', help="CUDA_DEVICE", required=False)

# Model parameter
parser.add_argument('--adgat_relation', help="W/WO Relation", action='store_true', required=False)
parser.add_argument('--gru_model', default='GRU', help="GRU, Graph_GRU, Graph_GRU_shared", required=False)
parser.add_argument('--number_of_layer', default=1, help="number of GRU layer", type=int, required=False)
parser.add_argument('--heads_att', default=6, help="the number of attention head", type=int, required=False)
parser.add_argument('--hidn_att', default=60, help="the hidden size of attention", type=int, required=False)
parser.add_argument('--hidden_size', default=64, help="the output hidden size of gru model", type=int, required=False)
parser.add_argument('--relation_static', default=None, help="relation_static", required=False)

# Hyperparameter
parser.add_argument('--batch_size', default=1, help="batch size", type=int, required=False)
parser.add_argument('--patience', default=20, help="patience", type=int, required=False)
parser.add_argument('--num_epochs', default=300, help="max epoch", type=int, required=False)
parser.add_argument('--learning_rate', default=1.00E-04, help="learning rate", type=float, required=False)
parser.add_argument('--weight_decay', default=1e-5, help="weight decay", type=float, required=False)
parser.add_argument('--dropout_rate', default=0.5, help="dropout rate", type=float, required=False)
parser.add_argument('--alpha', default=0.2, help="the slope of Leaky ReLU in Graph_attention model", type=float, required=False)

# Setup PyTorch Data Loader
class MyDataset(Dataset):
    def __init__(self, infos=None, features=None, labels=None, sub=None, device='cpu'):
        self._infos = infos
        self._features = features
        self._labels = labels
        self._sub = sub
        self._device = device

    def __getitem__(self, index):
        input_dict = {}
        input_dict['infos'] = self._infos[index]
        input_dict['features'] = torch.tensor(self._features[index], dtype = torch.float32).to(self._device)
        input_dict['labels'] = torch.tensor(self._labels[index], dtype = torch.double).to(self._device)

        # Check if sub exists
        if self._sub:
            input_dict['sub'] = torch.tensor(self._sub[index]).to(self._device)
        else:
            input_dict['sub'] = []
            
        return input_dict['infos'], input_dict['features'], input_dict['labels'], input_dict['sub']

    def __len__(self):
        return len(self._infos)


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, logits, labels):
        pad_event = torch.tensor(-1)
        not_pad = torch.ne(labels, pad_event)
        masks = not_pad.type(torch.float32)

        losses = (-1) * ( labels * torch.log(logits) + (1 - labels) * torch.log(1 - logits) )
        loss = torch.mean(losses * masks)
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

# Change_Training_Instance
def Change_Training_Instance(input_df, params, sampling, device):

    if sampling == True:
        all_company_path = "/tmp2/yhchen/Default_Prediction_Research_Project/explainable_credit/codes/data/interim/sample/all_company_id.csv"
    else:
        all_company_path = "/tmp2/cwlin/explainable_credit/data/all_company_ids.csv"
    all_company_df = pd.read_csv(all_company_path, header=0)
    all_company = all_company_df.id.sort_values().unique()

    date_group = input_df.groupby('date')

    # get train features
    features_, labels_ = [], []
    infos_ = []
    for date in input_df.date.sort_values().unique():
        df = date_group.get_group(date)
        # create rows with all companies fill with 0 if no data exists else fill with original data
        df1 = df.sort_values(by='id').set_index('id')

        df_new = pd.DataFrame(0, index=all_company, columns=df.columns)
        df_new.loc[df1.index, :] = df1
        df_new['id'] = df_new.index

        label_df = df_new.loc[:, ["y_cum_{:02d}".format(h) for h in range(1,1+8)]]
        label_df['y_cum_09'] = 1
        label_df.loc[label_df.index.difference(df1.index), :] = -1
        label = torch.tensor(label_df.values, dtype = torch.int).to(device)

        df_new.index = range(len(df_new))
        infos_.append([[date, i] for i in df_new.id])
        feature_window  = []
        for rnn_length in range(1, params.window_size+1):
            feature_df = df_new.loc[:, ['x_fea_{:02d}_w_{:02d}'.format(feat_i, rnn_length) for feat_i in range(1, params.feature_size+1)]]
            feature = torch.tensor(feature_df.values, dtype = torch.float32).to(device)
            feature_window.append(feature)
        
        feature_window = torch.stack(feature_window, dim=0)
        features_.append(feature_window) # 325 * (6, 15786, 14), (325, 15786, 14)
        labels_.append(label)         # 325 * (15786, 8)
    
    return infos_, features_, labels_

# Load Data
def load_dataset(path, params, sampling, device, is_training, split_valid):
    # Check the compression
    compression = "gzip" if ".gz" in path else None

    # Get infos, features, and labels (No for_column)
    # Read the data & skip the header
    df = pd.read_csv(path, compression=compression, header=0)

    # Fill missing values
    features_coverage = 2 + params.feature_size * params.window_size
    df.iloc[:, :2] = df.iloc[:, :2].fillna("") # info_df
    df.iloc[:, 2:features_coverage] = df.iloc[:, 2:features_coverage].fillna(0.0) # feature_df
    df.iloc[:, features_coverage:] = df.iloc[:, features_coverage:].fillna(0) # label_df

    # Replace other events as 0: 2 -> 0
    df.iloc[:, features_coverage:] = df.iloc[:, features_coverage:].replace(2, 0)

    infos_, features_, labels_ = Change_Training_Instance(df, params, sampling, device=device)

    if is_training:
        if split_valid:
            # Split train_set validation_set
            valid_size = 0.2
            num_train = len(features_)
            indices = list(range(num_train))
            np.random.shuffle(indices)
            split = int(np.floor(valid_size * num_train))
            train_idx, valid_idx = indices[split:], indices[:split]
        
            # define samplers for obtaining training and validation batches
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            dataset = MyDataset(infos=infos_, features=features_, labels=labels_, sub=None, device=device)
            train_loader = DataLoader(dataset, batch_size=params.batch_size, sampler=train_sampler)
            valid_loader = DataLoader(dataset, batch_size=params.batch_size, sampler=valid_sampler)
            return train_loader, valid_loader
        else:
            dataset = MyDataset(infos=infos_, features=features_, labels=labels_, sub=None, device=device)
            train_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=True)
            return train_loader
    else:
        dataset = MyDataset(infos=infos_, features=features_, labels=labels_, sub=None, device=device)
        test_loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False)
        return test_loader

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

def train(model, train_inputs, params, device):
    # Flag as training
    model.train()
    
    metrics = {
        'aucs': torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device),
        'loss': torch.empty((0), dtype=torch.float64).to(device),
        'batch_aucs': torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device),
        'batch_loss': torch.empty((0), dtype=torch.float64).to(device)
    }

    t = trange(len(train_inputs))
    for i, (infos, features, labels, subs) in zip(t, train_inputs):

        for j in range(params.batch_size):
            # Make predictions for this batch
            predict = model(features[j])
            
            # Compute the loss
            loss = criterion(predict, labels[j])

            # Calculate gradients
            loss.backward()

            batch_aucs = torch.tensor([[safe_auc(y_true=labels[j][:, i].cpu().detach().numpy(), 
                                y_score=predict[:, i].cpu().detach().numpy())
                                for i in range(params.cum_labels)]]).to(device)
            batch_loss = torch.tensor([loss.item()]).to(device)
            metrics['batch_aucs'] = torch.cat((metrics['batch_aucs'], batch_aucs), 0)
            metrics['batch_loss'] = torch.cat((metrics['batch_loss'], batch_loss), 0)  
        
        # Update weights
        optimizer.step()

        # Zero gradients for every batch
        optimizer.zero_grad()
        
        batch_aucs = torch.mean(metrics['batch_aucs'], 0, keepdim=True)
        batch_loss = torch.mean(metrics['batch_loss'], 0, keepdim=True)

        metrics['batch_aucs'] = torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device)
        metrics['batch_loss'] = torch.empty((0), dtype=torch.float64).to(device)

        metrics['aucs'] = torch.cat((metrics['aucs'], batch_aucs), 0)
        metrics['loss'] = torch.cat((metrics['loss'], batch_loss), 0)
            
        loss = batch_loss.item()

        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss))
            

    # Record training loss
    metrics['aucs'] = torch.mean(metrics['aucs'], 0)
    metrics['loss'] = torch.mean(metrics['loss'])

    k_aucs = ['auc_{}'.format(i+1) for i in range(params.cum_labels)]
    v_aucs = metrics['aucs'].tolist()
    metrics_val = dict(zip(k_aucs, v_aucs))
    metrics_val['auc'] = torch.mean(metrics['aucs']).item()
    metrics_val['loss'] = metrics['loss'].item()
    metrics_string = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_val.items())
    logging.info("- Train metrics: " + metrics_string)

# ----- Evaluation -----
def evaluate(model, eval_inputs, params, device):
    # Flag as evaluation
    model.eval()
    
    metrics = {
        'aucs': torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device),
        'loss': torch.empty((0), dtype=torch.float64).to(device),
        'batch_aucs': torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device),
        'batch_loss': torch.empty((0), dtype=torch.float64).to(device)
    }

    t = trange(len(eval_inputs))
    for i, (infos, features, labels, subs) in zip(t, eval_inputs):
        for j in range(params.batch_size):
            # Make predictions for this batch
            predict = model(features[j])
        
            # Compute the loss and its gradients
            loss = criterion(predict, labels[j])

            batch_aucs = torch.tensor([[safe_auc(y_true=labels[j][:, i].cpu().detach().numpy(), 
                                y_score=predict[:, i].cpu().detach().numpy())
                                for i in range(params.cum_labels)]]).to(device)
            batch_loss = torch.tensor([loss.item()]).to(device)
            metrics['batch_aucs'] = torch.cat((metrics['batch_aucs'], batch_aucs), 0)
            metrics['batch_loss'] = torch.cat((metrics['batch_loss'], batch_loss), 0)  
        
        batch_aucs = torch.mean(metrics['batch_aucs'], 0, keepdim=True)
        batch_loss = torch.mean(metrics['batch_loss'], 0, keepdim=True)

        metrics['batch_aucs'] = torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device)
        metrics['batch_loss'] = torch.empty((0), dtype=torch.float64).to(device)

        metrics['aucs'] = torch.cat((metrics['aucs'], batch_aucs), 0)
        metrics['loss'] = torch.cat((metrics['loss'], batch_loss), 0)
    
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

    # Set the random seed
    set_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()

    # setting for individual device allocating
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else args.device
    DEVICE = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    # Read the json file with parameters
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    """
        {
            For model parameter:
            {
                # In json
                "feature_size": 14,
                "window_size": 1,
                "model_version": "gru-type2",
                "layer_norm": "off",
                "batch_norm": "on",
                "label_type": "cum",
                "cum_labels": 8,
                "lstm_num_units": 64, == "hidden_size": 64
                "buffer_size": 1e6,

                # Not in json
                "relation_static": None,
                "adgat_relation": ,
                "gru_model": "GRU, Graph_GRU, Graph_GRU_shared",
                "number_of_layer": 1,
                "heads_att": 6,
                "hidn_att": 60
            }
            For hyperparameter:
            {
                # In json
                "learning_rate": 1e-4,
                "batch_size": 256,
                "num_epochs": 20,
                "dropout_rate": 0.5,
                "weight_decay": 1e-5,
                "save_summary_steps": 100

                # Not in json
                "patience": 20,
                "num_epochs": 300,
                "alpha": 0.2
            }
        }
    """

    # Params Implementation
    params.batch_size = 1

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")

    # Specify the names of the data files
    train_data = "train_{}.gz".format(params.label_type)
    test_data  = "test_{}.gz".format(params.label_type)

    # Paths to data
    train_data = os.path.join(args.data_dir, train_data)
    eval_data  = os.path.join(args.data_dir, test_data)

    # Skip header (Get the sizes)
    # params.train_size = int(os.popen('zcat ' + train_data + '|wc -l').read()) - 1
    # params.eval_size = int(os.popen('zcat ' + eval_data  + '|wc -l').read()) - 1
    # logging.info('Train size: {}, Eval size: {}'.format(params.train_size, params.eval_size))

    # cross-time or cross-section
    if "index" in train_data:
        print("Dataset : Cross-section")
    else:
        print("Dataset : Cross-time")

    # if sampling
    if "sample" in train_data:
        sampling = True
    else:
        sampling = False

    print("If sampling : ", sampling)

    # Load dataset
    logging.info("Loading dataset...")

    train_loader, valid_loader = load_dataset(path=train_data, params=params, sampling=sampling, device=DEVICE, is_training=True, split_valid=True)
    train_loader_all = load_dataset(path=train_data, params=params, sampling=sampling, device=DEVICE, is_training=True, split_valid=False)
    test_loader = load_dataset(path=eval_data, params=params, sampling=sampling, device=DEVICE, is_training=False, split_valid=False)

    # Params Implementation
    params.num_stock = len(next(iter(train_loader))[1][0][0])
    # params.num_stock = len(all_company)

    # hyper-parameters
    params.learning_rate = args.learning_rate
    params.dropout_rate = args.dropout_rate
    params.weight_decay = args.weight_decay
    params.patience = args.patience
    params.num_epochs = args.num_epochs
    params.alpha = args.alpha

    print("Hyperparameer Infomation : ")
    print("{")
    print("\tlearning rate : ", params.learning_rate)
    print("\tdropout rate : ", params.dropout_rate)
    print("\tweight decay : ", params.weight_decay)
    print("\tpatience : ", params.patience)
    print("\tnum_epochs : ", params.num_epochs)
    print("\talpha : ", params.alpha)
    print("}")

    # model setting
    params.adgat_relation = args.adgat_relation
    params.gru_model = args.gru_model
    params.number_of_layer = args.number_of_layer
    params.heads_att = args.heads_att
    params.hidn_att = args.hidn_att
    params.hidden_size = args.hidden_size
    params.relation_static = args.relation_static

    print("Creating model ...")
    print("Model Infomation : ")
    print("{")
    print("\tadgat relation : ", params.adgat_relation)
    print("\tgru model : ", params.gru_model)
    print("\tnumber of layer : ", params.number_of_layer)
    print("\theads att : ", params.heads_att)
    print("\thidn att : ", params.hidn_att)
    print("\thidden size : ", params.hidden_size)
    print("\trelation static : ", params.relation_static)
    print("}")

    model = Neural_Default_Prediction_revised(params=params, adgat_relation=args.adgat_relation)

    model.cuda(device=DEVICE)
    model.to(torch.float)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # criterion = torch.nn.NLLLoss()
    criterion = CustomCrossEntropyLoss()

    # Start training
    logging.info("Start training...")

    begin_at_epoch = 0
    best_eval_acc = 0.0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=params.patience, verbose=True)

    for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs): 
        # Train & Evaluate
        logging.info("Epoch {}/{}:".format(epoch + 1, begin_at_epoch + params.num_epochs))
        train(model, train_loader, params, DEVICE)
        valid_metrics = evaluate(model, valid_loader, params, DEVICE)

         # If best_eval, best_save_path
        eval_acc = valid_metrics['auc']
        if eval_acc >= best_eval_acc:
            # Store new best accuracy
            best_eval_acc = eval_acc
            # Save best eval metrics in a json file in the model directory
            best_json_path = os.path.join(args.model_dir, "metrics_eval_best_weights.json")
            save_dict_to_json(valid_metrics, best_json_path)

        # Save latest eval metrics in a json file in the model directory
        last_json_path = os.path.join(args.model_dir, "metrics_eval_last_weights.json")
        save_dict_to_json(valid_metrics, last_json_path)

        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_metrics['loss'], model)

        if early_stopping.early_stop:
            print("Early stopping")
            params.num_epochs = epoch + 1
            break

    # 2nd initialize model and train again
    model = Neural_Default_Prediction_revised(params=params, adgat_relation=args.adgat_relation)
    model.cuda(device=DEVICE)
    model.to(torch.float)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # criterion = torch.nn.NLLLoss()
    criterion = CustomCrossEntropyLoss()

    # Reset parameters
    model.reset_parameters()

    # Retrain model with fixed epoch
    for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
        # Train
        logging.info("Epoch {}/{}:".format(epoch + 1, begin_at_epoch + params.num_epochs))
        train(model, train_loader_all, params, DEVICE)

        # ret_metrics = evaluate(model, test_loader, params, DEVICE)


    # Save the final model
    final_save_path = os.path.join(args.model_dir, 'last_weights', 'checkpoint')
    torch.save(model, final_save_path)
