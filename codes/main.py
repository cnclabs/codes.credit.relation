import os
import time 
import logging
import argparse
import pandas as pd
import pickle
from tqdm import trange
import json

import torch
import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# import torch.utils.data as data
from torch.optim.lr_scheduler import ExponentialLR

from utils.config import DATA_DIR, EDGE_FILE_DIR, COMPANY_IDS
from utils.config import FEATURE_SIZE, WINDOW_SIZE, CUM_LABELS, NUM_COMPANIES
from utils.dataset_utils import load_dataset
from utils.loss_utils import CustomCrossEntropyLoss, safe_auc
from utils.model_utils import save_args_to_json, set_seed, set_logger
from models.models import NeuDP, CategoricalGraphAtt, CategoricalGraphAtt_wo_intra, CategoricalGraphAtt_wo_inter

def parse_args():
    parser = argparse.ArgumentParser()

    # experiments settings
    parser.add_argument('--model_name', choices=['NeuDP_GAT', 'NeuDP'], required=True)
    parser.add_argument('--device', default='1', help="CUDA_DEVICE")
    parser.add_argument('--experiment_type', required=True, choices=["index", "time", "expand_len"], help="dataset type")
    parser.add_argument('--fold', required=True, type=int, help="fold number of the data")

    # training settings
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float, help='1e-3, 1e-4, 1e-5')
    parser.add_argument('--weight_decay', default=0.000001, type=float, help='1e-4, 1e-5, 1e-6, l2 regularization')
    parser.add_argument('--gamma', default=0.9, type=float, help='exponential learning rate scheduler, lr = lr_0 * gamma ^ epoch')

    # model architecture settings
    parser.add_argument('--lstm_num_units', type=int, required=True)
    ## for NeuDP_GAT
    parser.add_argument('--cluster_setting', choices=['industry', 'kmeans'], default=None, help="company clustering method")
    parser.add_argument('--n_cluster', default=None, help="number of cluster, e.g. 100, ./data_dir/cluster_100, that contains inner/outer edges and company_to_sector_idx.pkl")
    parser.add_argument('--intra_gat_hidn_dim', type=int, default=None)
    parser.add_argument('--inter_gat_hidn_dim', type=int, default=None)

    return parser.parse_args()

def train(model, cum_labels, train_inputs, optimizer, criterion, device, relation_static = None):
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

def evaluate(model, cum_labels, eval_inputs, criterion, device, relation_static = None):
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

def create_experiment_directory(args, data_dir, window_size):
    experiment_type_mapping = {
        "index": "index",
        "time": "expand_no_overtime",
        "expand_len": "expand_len"
    }

    formatted_fold = f"{int(args.fold):02d}"

    data_dir = os.path.join(data_dir, experiment_type_mapping[args.experiment_type], f'time_fold_{formatted_fold}')

    model_dir = f"./experiments/{args.model_name}/{args.experiment_type}"
    if args.model_name == 'NeuDP_GAT':
        run_id = f"lstm{args.lstm_num_units}_intra{args.intra_gat_hidn_dim}_inter{args.inter_gat_hidn_dim}_lr{args.learning_rate}_wd{args.weight_decay}"
        model_dir = os.path.join(model_dir, f"{args.cluster_setting}_{args.n_cluster}", f"fold_{formatted_fold}", f"{args.model_name}_{window_size}_{args.experiment_type}_{run_id}")
        # model_dir = os.path.join(f"./experiments/{args.model_name}/{args.experiment_type}/{args.cluster_setting}_{args.n_cluster}/fold_{formatted_fold}/{args.model_name}_{window_size}_{args.experiment_type}_{run_id}")
    elif args.model_name == 'NeuDP':
        run_id = f"lstm{args.lstm_num_units}_lr{args.learning_rate}_wd{args.weight_decay}"
        model_dir = os.path.join(model_dir, f"fold_{formatted_fold}", f"{args.model_name}_{window_size}_{args.experiment_type}_{run_id}")
    
    # Create directories
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    return model_dir, data_dir

def initialize_model(args, num_company, device):
    if args.model_name == "NeuDP_GAT":
        logging.info("Loading inner edge, outer edge amd company sector...")
        inner_edge_idx = torch.load(os.path.join(EDGE_FILE_DIR, f'{args.cluster_setting}_cluster_{args.n_cluster}', 'inner_edge_idx.pt'), map_location=device)
        outer_edge_idx = torch.load(os.path.join(EDGE_FILE_DIR, f'{args.cluster_setting}_cluster_{args.n_cluster}', 'outer_edge_idx.pt'), map_location=device)
        with open(os.path.join(EDGE_FILE_DIR, f'{args.cluster_setting}_cluster_{args.n_cluster}', 'company_to_sector_idx.pkl'), 'rb') as f:
            company_to_sector_idx = pickle.load(f)
        
        model = CategoricalGraphAtt(FEATURE_SIZE, WINDOW_SIZE, CUM_LABELS+1, args.lstm_num_units, args.intra_gat_hidn_dim, args.inter_gat_hidn_dim, inner_edge_idx, outer_edge_idx, company_to_sector_idx, device)
        print("Model: NeuDP_GAT")

    elif args.model_name == "NeuDP_GAT_wo_intra":
        logging.info("Loading inner edge, outer edge amd company sector...")
        inner_edge_idx = torch.load(os.path.join(EDGE_FILE_DIR, f'{args.cluster_setting}_cluster_{args.n_cluster}', 'inner_edge_idx.pt'), map_location=device)
        outer_edge_idx = torch.load(os.path.join(EDGE_FILE_DIR, f'{args.cluster_setting}_cluster_{args.n_cluster}', 'outer_edge_idx.pt'), map_location=device)
        with open(os.path.join(EDGE_FILE_DIR, f'{args.cluster_setting}_cluster_{args.n_cluster}', 'company_to_sector_idx.pkl'), 'rb') as f:
            company_to_sector_idx = pickle.load(f)

        model = CategoricalGraphAtt_wo_intra(FEATURE_SIZE, WINDOW_SIZE, CUM_LABELS+1, args.lstm_num_units, args.intra_gat_hidn_dim, args.inter_gat_hidn_dim, inner_edge_idx, outer_edge_idx, company_to_sector_idx, device)
        print("Model: NeuDP_GAT_wo_intra")
    
    elif args.model_name == "NeuDP_GAT_wo_inter":
        logging.info("Loading inner edge, outer edge amd company sector...")
        inner_edge_idx = torch.load(os.path.join(EDGE_FILE_DIR, f'{args.cluster_setting}_cluster_{args.n_cluster}', 'inner_edge_idx.pt'), map_location=device)
        outer_edge_idx = torch.load(os.path.join(EDGE_FILE_DIR, f'{args.cluster_setting}_cluster_{args.n_cluster}', 'outer_edge_idx.pt'), map_location=device)
        with open(os.path.join(EDGE_FILE_DIR, f'{args.cluster_setting}_cluster_{args.n_cluster}', 'company_to_sector_idx.pkl'), 'rb') as f:
            company_to_sector_idx = pickle.load(f)

        model = CategoricalGraphAtt_wo_inter(FEATURE_SIZE, WINDOW_SIZE, CUM_LABELS+1, args.lstm_num_units, args.intra_gat_hidn_dim, args.inter_gat_hidn_dim, inner_edge_idx, outer_edge_idx, company_to_sector_idx, device)
        print("Model: NeuDP_GAT_wo_inter")
    
    elif args.model_name == "NeuDP":
        model = NeuDP(FEATURE_SIZE, WINDOW_SIZE, CUM_LABELS+1, args.lstm_num_units, num_company=num_company, device=device)
        print("Model: NeuDP")

    model.to(device)
    model.to(torch.float)

    return model

if __name__ == '__main__':
    args = parse_args()

    model_dir, data_dir = create_experiment_directory(args, DATA_DIR, WINDOW_SIZE)
    print(f"Model directory prepared at: {model_dir}")
    print(f"Data directory: {data_dir}")

    save_args_to_json(args, model_dir)

    set_seed() # Set the random seed
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    # Set the logger
    set_logger(os.path.join(model_dir, 'train.log'))

    # Args 
    logging.info(f"{args}")

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    logging.info("Loading all train dataset...")

    train_data = os.path.join(data_dir, 'train_cum.gz')
    start_t = time.time()
    train_all_inputs, _ = load_dataset(True, train_data, args.batch_size, FEATURE_SIZE, WINDOW_SIZE, COMPANY_IDS)
    end_t = time.time()
    logging.info("load all train dataset: {}s".format(end_t - start_t))

    model = initialize_model(args, NUM_COMPANIES, device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    criterion = CustomCrossEntropyLoss()


    # Start training all
    logging.info("Start training all...")
    final_num_epoch = args.num_epochs
    for epoch in range(final_num_epoch):
        # Train & Evaluate
        logging.info("Epoch {}/{}:".format(epoch + 1, final_num_epoch))
        train_ret_metric = train(model, CUM_LABELS, train_all_inputs, optimizer, criterion, device)
        scheduler.step()

    # Save the final model
    if not os.path.exists(f'{model_dir}/last_weights'):
        os.makedirs(f'{model_dir}/last_weights')
    # TODO: change the name of model
    final_save_path = os.path.join(model_dir, 'last_weights', f'{args.model_name}_checkpoint.pt')
    torch.save(model.state_dict(), final_save_path)
    logging.info(f"Model saved at {final_save_path}")