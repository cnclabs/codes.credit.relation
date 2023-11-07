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


# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
parser.add_argument('--label_type', default='cum', help="Label types, cum only or cum with forward label")
parser.add_argument('--restore_dir', default='None',
                    help="Optional, directory containing weights to reload before training")
parser.add_argument('--device', default='0', help="CUDA_DEVICE")


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


# Load Data
def load_dataset(is_training, filename, params, device):
    # Check the compression
    compression = "gzip" if ".gz" in filename else None

    # Get infos, features, and labels (No for_column)
    # Read the data & skip the header
    all_df = pd.read_csv(filename, compression=compression, header=0)
    
    # Input data format: date, id, x1 - x14 (depend on window size), y1 - y8 
    # Separate into infos, features, labels
    features_coverage = 2 + params.feature_size * params.window_size
    info_df, feature_df, label_df = all_df.iloc[:, :2], all_df.iloc[:, 2:features_coverage], all_df.iloc[:, features_coverage:]
    
    # Fill missing values
    info_df = info_df.fillna("")
    feature_df = feature_df.fillna(0.0)
    label_df = label_df.fillna(0)
    
    # Replace other events as 0: 2 -> 0
    label_df = label_df.replace(2, 0)
    
    # Convert to lists
    info_data, feature_data, label_data = info_df.values.tolist(), feature_df.values.tolist(), label_df.values.tolist()
    
    # Call PyTorch Data Loader
    dataset = MyDataset(infos=info_data, features=feature_data, labels=label_data, sub=None, device=device)
    iterator = DataLoader(dataset, batch_size=params.batch_size, shuffle=True if is_training else False)
    
    return iterator


# ----- Train -----
def train(model, train_inputs, params, device):
    # Flag as training
    model.train()
    
    metrics = {
        'aucs': torch.empty((0, int(params.cum_labels)), dtype=torch.float64).to(device),
        'loss': torch.empty((0), dtype=torch.float64).to(device)
    }

    t = trange(len(train_inputs))
    for i, (infos, features, labels, subs) in zip(t, train_inputs):
        # Zero gradients for every batch
        optimizer.zero_grad()
        
        # Make predictions for this batch
        predict = model(features)
        
        # Add an additional prediction horizon to represent the case that
        # every company will default in the infinite future
        # -> label (y) + 1: default
        labels = F.pad(labels, (0, 1), "constant", 1)
       
        # Compute the loss and its gradients
        loss = criterion(predict, labels)
        loss.backward()
        
        # Adjust learning weights
        optimizer.step()
    
        loss = loss.item()
        batch_aucs = torch.tensor([[safe_auc(y_true=labels[:, i].cpu().detach().numpy(), 
                            y_score=predict[:, i].cpu().detach().numpy())
                            for i in range(params.cum_labels)]]).to(device)
        batch_loss = torch.tensor([loss]).to(device)
        metrics['aucs'] = torch.cat((metrics['aucs'], batch_aucs), 0)
        metrics['loss'] = torch.cat((metrics['loss'], batch_loss), 0)
        
        # Log the loss in the tqdm progress bar
        t.set_postfix(loss='{:05.3f}'.format(loss))
    
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
        'loss': torch.empty((0), dtype=torch.float64).to(device)
    }
    
    t = trange(len(eval_inputs))
    for i, (infos, features, labels, subs) in zip(t, eval_inputs):
        # Make predictions for this batch
        predict = model(features)
        
        # label (y) + 1: default
        labels = F.pad(labels, (0, 1), "constant", 1)
        
        # Compute the loss and its gradients
        loss = criterion(predict, labels)
        loss = loss.item()
        
        batch_aucs = torch.tensor([[safe_auc(y_true=labels[:, i].cpu().detach().numpy(), 
                            y_score=predict[:, i].cpu().detach().numpy())
                            for i in range(params.cum_labels)]]).to(device)
        batch_loss = torch.tensor([loss]).to(device)
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
    params.train_size = int(os.popen('zcat ' + train_data + '|wc -l').read()) - 1
    params.eval_size = int(os.popen('zcat ' + eval_data  + '|wc -l').read()) - 1
    logging.info('Train size: {}, Eval size: {}'.format(params.train_size, params.eval_size))

    # Load dataset
    logging.info("Loading dataset...")
    train_inputs = load_dataset(True, train_data, params, DEVICE)
    # Evaluation -> Not training
    eval_inputs = load_dataset(False, eval_data, params, DEVICE)
    logging.info('Check train: {}, Check eval: {}'.format(len(train_inputs), len(eval_inputs)))
    logging.info("- done.")
    
    # Create the model
    logging.info("Creating the model...")
    model = Neural_Default_Prediction(params)
    
    if args.device != "cpu":
        model.cuda()
    logging.info("- done.")
    
    # Loss function
    criterion = CustomCrossEntropyLoss()
    
    # Adam Optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    
    # Start training
    logging.info("Start training...")
    
    begin_at_epoch = 0
    best_eval_acc = 0.0
    
    for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs): 
        # Train & Evaluate
        logging.info("Epoch {}/{}:".format(epoch + 1, begin_at_epoch + params.num_epochs))
        train(model, train_inputs, params, DEVICE)
        ret_metrics = evaluate(model, eval_inputs, params, DEVICE)
        
        # If best_eval, best_save_path
        eval_acc = ret_metrics['auc']
        if eval_acc >= best_eval_acc:
            # Store new best accuracy
            best_eval_acc = eval_acc
            # Save best eval metrics in a json file in the model directory
            best_json_path = os.path.join(args.model_dir, "metrics_eval_best_weights.json")
            save_dict_to_json(ret_metrics, best_json_path)

        # Save latest eval metrics in a json file in the model directory
        last_json_path = os.path.join(args.model_dir, "metrics_eval_last_weights.json")
        save_dict_to_json(ret_metrics, last_json_path)
    
    # Save the final model
    final_save_path = os.path.join(args.model_dir, 'last_weights', 'checkpoint')
    torch.save(model, final_save_path)
