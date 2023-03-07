from main import *
from prediction import prediction

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/processed/', help="Directory containing the dataset")
parser.add_argument('--data_file', default='test_cum.gz', help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")
parser.add_argument('--device', default='cpu', help="CUDA_DEVICE")


if __name__ == '__main__':
    # Set the random seed
    set_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()

    model_dir = args.model_dir # '/tmp2/cwlin/explainable_credit/explainable_credit/experiments/gru06_index/index_fold_06'
    data_dir = args.data_dir # '/tmp2/cwlin/explainable_credit/data/8_labels_index/len_06/index_fold_06'
    data_file = args.data_file # 'test_cum.gz'
    restore_dir = args.restore_dir # 'last_weights'
    device = args.device # '1'
    
    # setting for individual device allocating
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if device == 'cpu' else device
    DEVICE = torch.device("cuda" if device != "cpu" and torch.cuda.is_available() else "cpu")
    # DEVICE = torch.device(f"cuda:{device}")

    # Read the json file with parameters
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")

    # Paths to data
    eval_data  = os.path.join(data_dir, '{}'.format(data_file))

    # Skip header (Get the sizes)
    params.eval_size = int(os.popen('zcat ' + eval_data  + '|wc -l').read()) - 1

    params.batch_size = 1

    # if sampling
    if "sample" in eval_data:
        sampling = True
    else:
        sampling = False

    # Load dataset
    logging.info("Loading dataset...")
    eval_inputs = load_dataset(path=eval_data, params=params, sampling=sampling, device=DEVICE, is_training=False, split_valid=False)
    # with open('/tmp2/cwlin/default_prediction/codes/test2/x.pkl', 'rb') as f:
    #     x = pickle.load(f)
    # with open('/tmp2/cwlin/default_prediction/codes/test2/y.pkl', 'rb') as f:
    #     y = pickle.load(f)
    # with open('/tmp2/cwlin/default_prediction/codes/test2/results_dict.pkl', 'rb') as f:
    #     results_dict = pickle.load(f)

    # eval_inputs = x, y, results_dict

    logging.info("- done.")

    # Create the model
    # logging.info("Creating the model...")

    # model = Neural_Default_Prediction(params)
    # model = AD_GAT_without_relational(params=params, num_stock=NUM_STOCK,
    #                         d_hidden = D_MARKET, hidn_rnn = hidn_rnn, dropout = dropout) # 15786, 14

    # Test the model
    logging.info("Starting testing...")
    prediction(eval_inputs, model_dir, params, restore_dir, DEVICE)