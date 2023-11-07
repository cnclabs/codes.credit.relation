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

    compression = "GZIP" if ".gz" in filename else None

    _INFO_COLUMNS = ['date', 'id']
    _FEATURE_COLUMNS = ['x_fea_{:02d}_w_{:02d}'.format((j + 1), (i + 1))
            for i in range(params.window_size)
            for j in range(params.feature_size)]
    _LABEL_COLUMNS = ['y_cum_{:02d}'.format(i + 1) for i in range(params.cum_labels)]
    _FOR_COLUMNS = ['y_for_{:02d}'.format(i + 1) for i in range(60)]

    _INFO_DEFAULTS = ['' for i in range(len(_INFO_COLUMNS))]
    _FEATURE_DEFAULTS = [0.0 for i in range(len(_FEATURE_COLUMNS))]
    _LABEL_DEFAULTS = [0 for i in range(len(_LABEL_COLUMNS))]
    _FOR_DEFAULTS = [0 for i in range(len(_FOR_COLUMNS))]

    if "cum_for" in filename:
        _CSV_COLUMNS = _INFO_COLUMNS + _FEATURE_COLUMNS + _LABEL_COLUMNS + _FOR_COLUMNS
        _CSV_DEFAULTS = _INFO_DEFAULTS + _FEATURE_DEFAULTS + _LABEL_DEFAULTS + _FOR_DEFAULTS
    else:
        _CSV_COLUMNS = _INFO_COLUMNS + _FEATURE_COLUMNS + _LABEL_COLUMNS
        _CSV_DEFAULTS = _INFO_DEFAULTS + _FEATURE_DEFAULTS + _LABEL_DEFAULTS

    def parse_csv(line):
        print('Parsing')
        columns  = tf.decode_csv(line, record_defaults=_CSV_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))

        if "cum_for" in filename:
            sub = [features.pop(s) for s in _FOR_COLUMNS]
            infos    = [features.pop(i) for i in _INFO_COLUMNS]
            labels   = [features.pop(l) for l in _LABEL_COLUMNS]
            features = [features.pop(f) for f in _FEATURE_COLUMNS]

            return features, labels, infos, sub

        else:
            infos    = [features.pop(i) for i in _INFO_COLUMNS]
            labels   = [features.pop(l) for l in _LABEL_COLUMNS]
            features = [features.pop(f) for f in _FEATURE_COLUMNS]

            return features, labels, infos


    if is_training:
        dataset = (tf.data
                   .TextLineDataset(filename,
                       compression_type=compression)
                   .skip(1)
                   .shuffle(buffer_size=int(params.train_size))
                   .map(parse_csv, num_parallel_calls=10)
                   .batch(params.batch_size)
                   .prefetch(1))  # always have 1 batch ready to serve
    else:
        dataset = (tf.data
                   .TextLineDataset(filename,
                       compression_type=compression)
                   .skip(1)
                   .map(parse_csv, num_parallel_calls=10)
                   .batch(params.batch_size)
                   .prefetch(1))  # always have 1 batch ready to serve

    iterator = dataset.make_initializable_iterator()

    if "cum_for" in filename:
        features, labels, infos, sub_labels = iterator.get_next()
    else:
        features, labels, infos = iterator.get_next()

    iterator_init_op = iterator.initializer
    inputs = {'features': features,
              'labels': labels,
              'infos': infos,
              'iterator_init_op': iterator_init_op}

    if "cum_for" in filename:
        inputs['sub_labels'] = sub_labels

    return inputs
