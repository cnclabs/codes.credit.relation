"""Tensorflow utility functions for training"""

import logging
import os

import tensorflow as tf
import pandas as pd
import numpy as np

from tqdm import trange


def evaluate_sess(sess, model_dir, model_spec, num_steps, writer=None, params=None):
    """Train the model on `num_steps` batches.

    Args:
        sess: (tf.Session) current session
        model_spec: (dict) contains the graph operations or nodes needed for training
        num_steps: (int) train for this number of batches
        writer: (tf.summary.FileWriter) writer for summaries. Is None if we don't log anything
        params: (Params) hyperparameters
    """
    data = [model_spec['predictions'],
            model_spec['infos'],
            model_spec['labels'],
            model_spec['num_paras']]
    update_metrics = model_spec['update_metrics']
    eval_metrics = model_spec['metrics']
    global_step = tf.train.get_global_step()

    # Load the evaluation dataset into the pipeline and initialize the metrics init op
    sess.run(model_spec['metrics_init_op'])

    # write file
    predict_path = os.path.join(model_dir, 'pred.csv')
    header_str = 'date,id,' + ','.join(
            "p_cum_{:02d}".format(v + 1) for v in range(params.cum_labels)) + '\n'

    with open(predict_path, 'w') as f:
        f.write(header_str)

    # compute metrics over the dataset
    sess.run(model_spec['iterator_init_op'])
    for _ in trange(num_steps):
        pred, info, label, paras = sess.run(data)
        print(paras)
        result = np.concatenate((info, pred), axis=1)
        with open(predict_path,'ab') as f:
            np.savetxt(f, result, delimiter=',', fmt='%s')

    logging.info("Pred done, saved in {}".format(predict_path))


def prediction(model_spec, model_dir, params, restore_from):
    """Evaluate the model

    Args:
        model_spec: (dict) contains the graph operations or nodes needed for evaluation
        model_dir: (string) directory containing config, weights and log
        params: (Params) contains hyperparameters of the model.
                Must define: num_epochs, train_size, batch_size, eval_size, save_summary_steps
        restore_from: (string) directory or file containing weights to restore the graph
    """
    # Initialize tf.Saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Initialize the lookup table
        sess.run(model_spec['variable_init_op'])

        # Reload weights from the weights subdirectory
        save_path = os.path.join(model_dir, restore_from)
        if os.path.isdir(save_path):
            save_path = tf.train.latest_checkpoint(save_path)
        saver.restore(sess, save_path)

        # Evaluate
        num_steps = (params.eval_size + params.batch_size - 1) // params.batch_size
        evaluate_sess(sess, model_dir, model_spec, num_steps, params=params)
