from main import *
from prediction import prediction


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/processed/', help="Directory containing the dataset")
parser.add_argument('--data_file', default='test_cum.gz', help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")
parser.add_argument('--device', default='0', help="CUDA_DEVICE")


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

    # Paths to data
    eval_data  = os.path.join(args.data_dir, '{}'.format(args.data_file))
    
    # Skip header (Get the sizes)
    params.eval_size = int(os.popen('zcat ' + eval_data  + '|wc -l').read()) - 1

    # Load dataset
    logging.info("Loading dataset...")
    eval_inputs = load_dataset(False, eval_data, params, DEVICE)
    logging.info("- done.")
    
    # Create the model
    logging.info("Creating the model...")
    model = Neural_Default_Prediction(params)
    
    # Test the model
    logging.info("Starting testing...")
    prediction(eval_inputs, args.model_dir, params, args.restore_dir, DEVICE)
