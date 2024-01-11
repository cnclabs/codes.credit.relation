"""General utility functions"""

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


# def save_dict_to_json(d, json_path, mode='w', h_params=None):
#     """Saves dict of floats in json file

#     Args:
#         d: (dict) of float-castable values (np.float, int, float, etc.)
#         json_path: (string) path to json file
#         mode: originally is 'w', but searching hyperparameter I add 'a' for append
#         h_params: (dict) of hyperparameters
#     """
#     with open(json_path, mode=mode) as f:
#         if h_params is not None:
#             ## cwlin
#             d = {k: float(v) for k, v in d.items()}
#             json.dump([{"h_params": h_params, "performance": d}], f, indent=4)
#         else:
#             ## orginial
#             # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
#             d = {k: float(v) for k, v in d.items()}
#             json.dump(d, f, indent=4)


import os

def save_dict_to_json(d, json_path, h_params=None):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
    d = {k: float(v) for k, v in d.items()}
    
    if not os.path.isfile(json_path):
        with open(json_path, 'w') as f:
            json.dump([{"h_params": h_params, "performance": d}], f, indent=4)
    else:
        with open(json_path, 'r+') as f:
            f.seek(0, os.SEEK_END)  # Go to the end of file    
            f.seek(f.tell() - 2, os.SEEK_SET)  # Move 2 character before the end
            f.truncate()  # Remove the last 2 characters
            f.write(',\n')  # Write a comma and newline
            json.dump({"h_params": h_params, "performance": d}, f, indent=4)  # Dump the dictionary
            f.write('\n]')  # Write a closing bracket