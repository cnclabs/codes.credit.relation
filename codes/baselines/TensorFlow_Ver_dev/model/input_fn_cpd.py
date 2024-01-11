"""
Input pipeline for dataset API
"""

import tensorflow as tf


def input_fn(is_training, filename, params):
    """
    Args:
        is_training: (bool) whether to use the train or test pipeline.
                     At training, we shuffle the data and have multiple epochs
        filenames: (list) filenames of the data (csv format)
        params: (Params) contains hyperparameters of the model
                (ex: `params.num_epochs`)
    """
    _INFO_COLUMNS = ['date', 'id']
    _FEATURE_COLUMNS = ['x_fea_{:02d}'.format(i+1) for i in range(14)]
    _LABEL_COLUMNS = ['y_cum_{:02d}'.format(i+1) for i in range(6)]
    _CPD_COLUMNS = ['x_cpd_{:02d}'.format(i+1) for i in range(6)]
    _CSV_COLUMNS = _INFO_COLUMNS + _FEATURE_COLUMNS + _LABEL_COLUMNS + _CPD_COLUMNS
    _INFO_DEFAULTS = ['' for i in range(len(_INFO_COLUMNS))]
    _FEATURE_DEFAULTS = [0.0 for i in range(len(_FEATURE_COLUMNS))]
    _LABEL_DEFAULTS = [0 for i in range(len(_LABEL_COLUMNS))]
    _CPD_DEFAULTS = [0.0 for i in range(len(_CPD_COLUMNS))]
    _CSV_DEFAULTS = _INFO_DEFAULTS + _FEATURE_DEFAULTS + _LABEL_DEFAULTS + _CPD_DEFAULTS

    def parse_csv(line):
        print('Parsing')
        columns  = tf.decode_csv(line, record_defaults=_CSV_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        cpds =     [features.pop(c) for c in _CPD_COLUMNS]
        labels   = [features.pop(l) for l in _LABEL_COLUMNS]
        infos    = [features.pop(i) for i in _INFO_COLUMNS]
        features = [features.pop(f) for f in _FEATURE_COLUMNS]
        return features, labels, infos, cpds
    
    if is_training:
        dataset = (tf.data.TextLineDataset(filename)
                   .skip(1)
                   .shuffle(10000)
                   .map(parse_csv, num_parallel_calls=10)
                   .batch(params.batch_size)
                   .prefetch(1))  # always have 1 batch ready to serve
    else:
        dataset = (tf.data.TextLineDataset(filename)
                   .skip(1)
                   .map(parse_csv, num_parallel_calls=10)
                   .batch(params.batch_size)
                   .prefetch(1))  # always have 1 batch ready to serve

    iterator = dataset.make_initializable_iterator()
    features, labels, infos, cpds = iterator.get_next()
    iterator_init_op = iterator.initializer
    inputs = {'features': features,
              'labels': labels,
              'infos': infos,
              'cpds': cpds,
              'iterator_init_op': iterator_init_op}
    
    return inputs
