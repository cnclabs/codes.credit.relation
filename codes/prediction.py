"""Tensorflow utility functions for training"""
import logging
import os
import pandas as pd
import numpy as np
from tqdm import trange
import torch


def evaluate_sess(model, dataloader, results_dict, model_dir, params=None):
    """Train the model on `num_steps` batches."""
    # write file
    predict_path = os.path.join(model_dir, 'pred.csv')
    header_str = 'date,id,' + ','.join(
            "p_cum_{:02d}".format(v + 1) for v in range(params.cum_labels)) + '\n'

    with open(predict_path, 'w') as f:
        f.write(header_str)

    t = trange(len(dataloader))

    temp = iter(results_dict)

    for i, (features, labels), (key) in zip(t, dataloader, temp):
        batch_size = features.size(0)
        for batch in range(batch_size):
            if batch!=0:
                key = next(temp, None)
                if key!=None:
                    df = results_dict[key]
            else:
                df = results_dict[key]

            model.eval()
            # Predictions & loss
            predict = model(features[batch])
            df_pred = pd.DataFrame(predict.detach().cpu().numpy())
            result = df.join(df_pred)
            result.to_csv(predict_path, mode='a', header=None, index=None)

    logging.info("Pred done, saved in {}".format(predict_path))


def prediction(dataloader, results_dict, model_dir, params, restore_from, device):
    """Evaluate the model"""

    # Reload weights from the weights subdirectory
    save_path = os.path.join(model_dir, restore_from)
    if os.path.isdir(save_path):
        model_path = os.path.join(save_path, 'checkpoint')
        model = torch.load(model_path, map_location=torch.device('cuda'))
        model.to(device)
        
    evaluate_sess(model, dataloader, results_dict, model_dir, params=params)
