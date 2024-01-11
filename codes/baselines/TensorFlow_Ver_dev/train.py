"""Train the model"""

import argparse
import logging
import os

import tensorflow as tf

from model.utils import Params
from model.utils import set_logger
from model.training import train_and_evaluate
from model.input_fn import input_fn
from model.model_fn import model_fn

## cwlin
import math

# Test
# from modified_model import model_fn
#from model.model_fn_forward import model_fn


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--data_dir', default='data/', help="Directory containing the dataset")
parser.add_argument('--label_type', default='cum', help="Label types, cum only or cum with forward label")
parser.add_argument('--restore_dir', default='None',
                    help="Optional, directory containing weights to reload before training")
parser.add_argument('--device', default='0', help="CUDA_DEVICE")

## cwlin
# training settings
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--learning_rate', default=1e-2, type=float, help='1e-2, 1e-3, 1e-4, 1e-5')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='1e-4, 1e-5, 1e-6, l2 regularization')
parser.add_argument('--gamma', default=0.9, type=float, help='exponential learning rate scheduler, lr = lr_0 * gamma ^ epoch')

if __name__ == '__main__':

    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from the experiment params.json file in model_dir
    args = parser.parse_args()

    # setting for individual device allocating
    os.environ['CUDA_VISIBLE_DEVICES'] = '' if args.device == 'cpu' else args.device

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)
    ## cwlin: overwrite the params
    params.num_epochs = args.num_epochs
    params.batch_size = args.batch_size
    params.learning_rate = args.learning_rate
    params.weight_decay = args.weight_decay
    params.decay_rate = args.gamma
    params.validation = False

    ## Check that we are not overwriting some previous experiment
    ## Comment these lines if you are developing your model and don't care about overwritting
    #model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "best_weights"))
    #overwritting = model_dir_has_best_weights and args.restore_dir is None
    #assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")

    ## cwlin: train_subset and valid_subset are used for the early stopping epoch and patience, then with the determined epoch, train again and test again
    train_data = "train_{}.gz".format(params.label_type)
    test_data  =  "test_{}.gz".format(params.label_type)

    # Paths to data
    train_data = os.path.join(args.data_dir, train_data)
    eval_data  = os.path.join(args.data_dir, test_data)
    # skip header
    params.train_size = int(os.popen('zcat ' + train_data + '|wc -l').read()) - 1
    params.eval_size = int(os.popen('zcat ' + eval_data  + '|wc -l').read()) - 1
    logging.info('Train size: {}, Eval size: {}'.format(params.train_size, params.eval_size))

    # cwlin: For setting the decay_steps to the number of batches in an epoch. This way, the learning rate would be decayed once per epoch, just like in PyTorch.
    num_batches_per_epoch = math.ceil(params.train_size / params.batch_size)
    params.decay_steps = int(num_batches_per_epoch * params.num_epochs)

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_data, params)
    eval_inputs = input_fn(False, eval_data, params)
    logging.info("- done.")

    # Define the models (2 different set of nodes that share weights for train and eval)
    logging.info("Creating the model...")
    train_model_spec = model_fn(True, train_inputs, params)
    eval_model_spec = model_fn(False, eval_inputs, params, reuse=True)
    logging.info("- done.")

    if args.restore_dir != "None":
        restore_dir = os.path.join(args.model_dir, args.restore_dir)
    else:
        restore_dir = None
    
    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec, args.model_dir, params, restore_dir)
    