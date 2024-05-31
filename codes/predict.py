import os 
import time
import logging
import argparse
import pandas as pd
import pickle
import torch

from utils.config import DATA_DIR, COMPANY_IDS
from utils.config import FEATURE_SIZE, WINDOW_SIZE, CUM_LABELS, NUM_COMPANIES
from utils.dataset_utils import load_dataset
from utils.model_utils import set_seed, set_logger
from models.models import NeuDP, CategoricalGraphAtt, CategoricalGraphAtt_wo_intra, CategoricalGraphAtt_wo_inter

from main import create_experiment_directory, initialize_model
from prediction import prediction


def parse_args():
    parser = argparse.ArgumentParser()
    # experiments settings
    parser.add_argument('--model_name', choices=['NeuDP_GAT', 'NeuDP'], required=True)
    parser.add_argument('--device', default='cpu', help="CUDA_DEVICE")
    parser.add_argument('--experiment_type', required=True, choices=["index", "time", "expand_len"], help="dataset type")
    parser.add_argument('--fold', required=True, type=int, help="fold number of the data")

    # data settings
    parser.add_argument('--data_file', default='test_cum.gz', help="valid_subset_cum.gz, test_cum.gz")
    parser.add_argument('--restore_dir', default='last_weights', help="Optional, directory containing weights to reload before training")

    # training settings
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

if __name__ == '__main__':
    args = parse_args()

    model_dir, data_dir = create_experiment_directory(args, DATA_DIR, WINDOW_SIZE)

    # Set the random seed
    set_seed() 

    # setting for individual device allocating
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else args.device
    device = torch.device("cuda" if args.device != "cpu" and torch.cuda.is_available() else "cpu")

    # Set the logger
    set_logger(os.path.join(model_dir, 'train.log'))
    
    # data_file = args.data_file
    # restore_dir = args.restore_dir


    # Create the input data pipeline
    logging.info("Creating the datasets...")

    # Paths to data
    eval_data  = os.path.join(data_dir, '{}'.format(args.data_file))
    file_type = args.data_file.split('_')[0] # test / valid

    # Load dataset
    logging.info(f"Loading {file_type} dataset...")
    # all_company = pd.read_csv(ALL_COMPANY_IDS_PATH, index_col=0).id.sort_values().unique()
    # num_company = len(all_company) # 15786

    start_t = time.time()
    eval_inputs, results_dict = load_dataset(False, eval_data, args.batch_size, FEATURE_SIZE, WINDOW_SIZE, COMPANY_IDS)
    end_t = time.time()
    print(f'time for loading {file_type} dataset: ', end_t - start_t)

    logging.info("- done.")

    # Initialize the model
    model = initialize_model(args, NUM_COMPANIES, device)

    save_path = os.path.join(model_dir, args.restore_dir)
    if os.path.isdir(save_path):
        model_path = os.path.join(save_path, f'{args.model_name}_checkpoint.pt')
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        print('using:', model.device)
    
    # Test the model
    logging.info("Starting testing...")
    prediction(model, eval_inputs, results_dict, model_dir, args.model_name, device, CUM_LABELS, file_type)