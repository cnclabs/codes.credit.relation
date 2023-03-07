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
        infos = np.array(infos, dtype=list)
        # infos: (number of company, date & id, batch size)
        # predict: (number of company, cum)
        for batch in range(features.shape[0]):
            result = []
            # Predictions & loss
            predict = model(features[batch])
            for id in range(len(infos)):
                info_csv = []
                info_csv.append(infos[id][0][batch])
                info_csv.append(infos[id][1][batch].tolist())
                info_csv = info_csv + predict[id].tolist()
                result.append(info_csv)
                # result = np.concatenate((info_csv, predict[id].detach().numpy()))
            result = np.array(result)
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
