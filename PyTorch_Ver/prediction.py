"""Tensorflow utility functions for training"""
import logging
import os
import pandas as pd
import numpy as np
from tqdm import trange
import torch


def evaluate_sess(model, dataloader, model_dir, params=None):
    """Train the model on `num_steps` batches."""
    # write file
    predict_path = os.path.join(model_dir, 'pred.csv')
    header_str = 'date,id,' + ','.join(
            "p_cum_{:02d}".format(v + 1) for v in range(params.cum_labels)) + '\n'

    with open(predict_path, 'w') as f:
        f.write(header_str)

    t = trange(len(dataloader))
    for i, (infos, features, labels, subs) in zip(t, dataloader):
        model.eval()
        # Predictions & loss
        predict = model(features)
        
        info_csv = list(zip(infos[0], infos[1].detach().numpy()))
        result = np.concatenate((info_csv, predict.detach().numpy()), axis=1)
        with open(predict_path,'ab') as f:
            np.savetxt(f, result, delimiter=',', fmt='%s')

    logging.info("Pred done, saved in {}".format(predict_path))


def prediction(dataloader, model_dir, params, restore_from, device):
    """Evaluate the model"""

    # Reload weights from the weights subdirectory
    save_path = os.path.join(model_dir, restore_from)
    if os.path.isdir(save_path):
        model_path = os.path.join(save_path, 'checkpoint')
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.to(device)
        
    evaluate_sess(model, dataloader, model_dir, params=params)
