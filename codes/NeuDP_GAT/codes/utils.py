from torch import nn
import math
import random
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import json
import logging


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def createPath(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

def metrics(trues, preds):
    trues = np.concatenate(trues,-1)
    preds = np.concatenate(preds,0)
    acc = sum(preds.argmax(-1) == trues) / len(trues)
    auc = roc_auc_score(trues,preds[:,1])
    return acc, auc

# Utility function to save arguments to a JSON file
def save_args_to_json(args, json_file_path):
    """
    Save command line arguments to a JSON file.

    Args:
        args (Namespace): Parsed command line arguments.
        json_file_path (str): Path to the JSON file.
    """

    # Convert args namespace to dictionary
    args_dict = vars(args)

    # Ensure json_file_path is a file, not a directory
    if json_file_path.endswith('/'):
        json_file_path = json_file_path.rstrip('/')  # remove trailing slash if any
    if not json_file_path.endswith('.json'):
        json_file_path = f"{json_file_path}/args.json"  # append filename to directory


    with open(json_file_path, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

# Function to print model with hierarchy and parameters
def print_model_with_hierarchy(model, indent=""):
    total_params = 0
    for name, module in model.named_children():
        print(f"{indent}{name}:")
        if list(module.children()):
            child_params = print_model_with_hierarchy(module, indent + "  ")
            total_params += child_params
        else:
            for param_name, param in module.named_parameters():
                if param.requires_grad:
                    param_size = param.numel()
                    total_params += param_size
                    print(f"{indent}  {param_name}: Parameters: {param_size}")
    return total_params

def write_model_hierarchy_to_file(model, file_path, indent=""):
    total_params = 0
    with open(file_path, 'w') as file:
        def write_hierarchy(model, indent=""):
            nonlocal total_params
            for name, module in model.named_children():
                file.write(f"{indent}{name}:\n")
                if list(module.children()):
                    child_params = write_hierarchy(module, indent + " ")
                    total_params += child_params
                else:
                    for param_name, param in module.named_parameters():
                        if param.requires_grad:
                            param_size = param.numel()
                            total_params += param_size
                            file.write(f"{indent}  {param_name}: Parameters: {param_size}\n")
            return total_params
        write_hierarchy(model)
    
    return total_params