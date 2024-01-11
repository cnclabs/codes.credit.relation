"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.predict import prediction 
from model.input_fn import input_fn
from model.model_fn import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/processed/', help="Directory containing the dataset")
parser.add_argument('--data_file', default='test_cum.gz', help="Directory containing the dataset")
parser.add_argument('--restore_dir', default=None,
                    help="Optional, directory containing weights to reload before training")
parser.add_argument('--device', default='0',
                    help="CUDA_VISIBLE_DEVICES")


if __name__ == '__main__':
    args = parser.parse_args()
    file_type = args.data_file.split('_')[0] # test / valid

    # setting for individual device allocating
    os.environ['CUDA_VISIBLE_DEVICES'] = "" if args.device == "cpu" else args.device

    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")

    # Paths to data
    eval_data = os.path.join(args.data_dir, '{}'.format(args.data_file))
    params.eval_size = int(os.popen('zcat ' + eval_data + '|wc -l').read()) - 1

    # Create the two iterators over the two datasets
    eval_inputs = input_fn(False, eval_data, params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    model_spec = model_fn(False, eval_inputs, params)
    logging.info("- done.")

    # Train the model
    logging.info("Starting evaluation".format(params.num_epochs))
    prediction(model_spec, args.model_dir, params, args.restore_dir, file_type=file_type)
