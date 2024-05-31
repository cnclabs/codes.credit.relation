"""Tensorflow utility functions for training"""
import logging
import os
import pandas as pd
import numpy as np
from tqdm import trange
import torch


def evaluate_sess(model, dataloader, results_dict, model_dir, model_name, device, cum_labels=8, file_type='test'):
    """Train the model on `num_steps` batches."""
    # write file
    predict_path = os.path.join(model_dir, f'{model_name}_pred_{file_type}.csv')
    header_str = 'date,id,' + ','.join(
            "p_cum_{:02d}".format(v + 1) for v in range(cum_labels)) + '\n'

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
            inputs = features[batch].to(device)
            predict = model(inputs)
            df_pred = pd.DataFrame(predict.detach().cpu().numpy())
            result = df.join(df_pred)
            result.to_csv(predict_path, mode='a', header=None, index=None)
            
            # info_csv = list(zip(infos[0], infos[1].detach().numpy()))
            # result = np.concatenate((info_csv, predict.detach().numpy()), axis=1)
            # with open(predict_path,'ab') as f:
                # np.savetxt(f, result, delimiter=',', fmt='%s')
        
    logging.info("Pred done, saved in {}".format(predict_path))


def prediction(model, dataloader, results_dict, model_dir, model_name, device, cum_labels, file_type):
    """Evaluate the model"""

    # Reload weights from the weights subdirectory
    # save_path = os.path.join(model_dir, restore_from)
    # if os.path.isdir(save_path):
    #     model_path = os.path.join(save_path, f'{model_name}_checkpoint_{n_cluster}')
    #     model = torch.load(model_path, map_location=device)
    #     model.to(device)
        
    evaluate_sess(model, dataloader, results_dict, model_dir, model_name, device, cum_labels, file_type=file_type)
