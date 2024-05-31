from main import *
from prediction import prediction
import time

parser = argparse.ArgumentParser()
# experiments settings
parser.add_argument('--model_dir', default='./experiments/gru12_index')
parser.add_argument('--model_name', default='NeuDP_GAT')
parser.add_argument('--data_dir', default='/home/cwlin/explainable_credit/data', help="Directory containing the dataset")
parser.add_argument('--device', default='cpu', help="CUDA_DEVICE")
parser.add_argument('--all_company_ids_path', default='/home/cwlin/explainable_credit/data/all_company_ids.csv', help="path to list of all company ids")
parser.add_argument('--feature_size', default=14, type=int)
parser.add_argument('--window_size', default=12, type=int)
parser.add_argument('--cum_labels', default=8, type=int)

# data settings
parser.add_argument('--data_file', default='test_cum.csv', help="valid_subset_cum.csv, test_cum.csv")
parser.add_argument('--restore_dir', default='last_weights',
                    help="Optional, directory containing weights to reload before training")

# model architecture settings
parser.add_argument('--lstm_num_units', default=64, type=int)
parser.add_argument('--batch_size', default=1, type=int)

if __name__ == '__main__':
    # Set the random seed
    set_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()

    model_dir = args.model_dir 
    model_name = args.model_name
    data_dir = args.data_dir
    data_file = args.data_file
    restore_dir = args.restore_dir
    cum_labels = args.cum_labels
    device = args.device
    all_company_ids_path = args.all_company_ids_path

    # setting for individual device allocating
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if device == 'cpu' else device
    device = torch.device(f"cuda:{device}" if device != "cpu" and torch.cuda.is_available() else "cpu")

    # Set the logger
    set_logger(os.path.join(model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")

    # Paths to data
    eval_data  = os.path.join(data_dir, '{}'.format(data_file))
    file_type = data_file.split('_')[0] # test / valid

    # Skip header (Get the sizes)
    # params.eval_size = int(os.popen('zcat ' + eval_data  + '|wc -l').read()) - 1

    # Load dataset
    logging.info("Loading {} dataset...".format(file_type))
    all_company_ids = pd.read_csv(f'{all_company_ids_path}', index_col=0).id.sort_values().unique()
    start_t = time.time()
    eval_inputs, results_dict = load_dataset(False, eval_data, args.batch_size, args.feature_size, args.window_size, all_company_ids)
    end_t = time.time()
    print(f'time for loading {file_type} dataset: ', end_t - start_t)

    logging.info("- done.")

    # re-initialize model
    model = NeuDP(args.feature_size, args.window_size, cum_labels+1, args.lstm_num_units, num_company=len(all_company_ids), device=device)
    
    # File path for saving the model hierarchy
    model_hierarchy_path = os.path.join(model_dir, 'model_hierarchy.txt')

    # Call the modified function to write the model hierarchy to the file
    total_params = write_model_hierarchy_to_file(model, model_hierarchy_path)
    print(f"Total model parameters: {total_params}")

    # You can also write the total number of parameters to the file if needed
    with open(model_hierarchy_path, 'a') as f:
        f.write(f"\nTotal model parameters: {total_params}\n")
    
    save_path = os.path.join(model_dir, restore_dir)
    if os.path.isdir(save_path):
        model_path = os.path.join(save_path, f'train_valid_{model_name}_checkpoint.pt')
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        print('using:', model.device)
    
    # Test the model
    logging.info("Starting testing...")
    prediction(model, eval_inputs, results_dict, model_dir, model_name, device, cum_labels, file_type)