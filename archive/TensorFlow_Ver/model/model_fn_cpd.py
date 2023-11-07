"""
Tensorflow graph(node/ops) definitions

"""
import tensorflow as tf
import model.custom_cell as custom_cell
from tensorflow.keras.constraints import NonNeg


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)
    Args:
        is_training: (bool)
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model
                (ex: `params.learning_rate`)
    Returns:
        output: (tf.Tensor) output of the model
    """
    sentence = tf.reshape(inputs['features'],
                          [-1, params.feature_size, params.window_size])
    sentence = tf.transpose(sentence, [0, 2, 1])
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(params.lstm_num_units)
    output, final_state = tf.nn.dynamic_rnn(
        cell=lstm_cell, inputs=sentence,
        dtype=tf.float32
    )
    # fully connected layer at the last node
    # can be treated as summary vector
    logits = tf.layers.dense(inputs=final_state.h, units=7)
    cpds = tf.pad(inputs['cpds'], [[0, 0], [0, 1]], constant_values=1.0)
    logits = logits + cpds
    logits = tf.nn.softmax(logits, axis=1)
    logits = tf.cumsum(logits, axis=1)
    logits = logits[:, :-1]
    eps = 1e-8
    logits = tf.clip_by_value(logits, eps, 1-eps)

    return logits


def model_fn(is_training, inputs, params, reuse=False):
    """Model function defining the graph operations.
    Args:
        is_training: (bool)
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights
    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    # To replace other events as 0
    default_event = tf.constant(1, shape=None)
    labels = inputs['labels'][:, :]
    comparison = tf.equal(labels, default_event)
    labels = tf.where(comparison, labels, tf.zeros_like(labels))
    labels = tf.cast(labels, dtype=tf.float32)

    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        # Return the normalized value w.r.t. each logit
        predictions = logits

    
    # Define loss and accuracy (we need to apply a mask to account for padding)
    #losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
    #                                                 labels=labels)
    #losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
    #                                                 labels=labels)
    # cross-entropy
    losses = ((-1) * labels * tf.log(logits)
            - (1 - labels) * tf.log(1- logits))
    ## PoissonNLL
    #losses = logits - labels * tf.log(logits)

    loss = tf.reduce_mean(losses)
    
    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)
        
    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = dict(zip(['auc_' + str(i) for i in range(6)],
                           [tf.metrics.auc(labels=labels[:, i],
                                  predictions=predictions[:, i])
                            for i in range(6)]))
        metrics = {'auc': tf.metrics.mean(
            tf.reduce_mean([tf.metrics.auc(
                labels=labels[:, i], predictions=predictions[:, i]) 
                for i in range(6)])
        )}
        metrics['loss'] = tf.metrics.mean(loss)
    
    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])
    
    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                         scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    #tf.summary.scalar('auc', auc_score)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    variable_init_op = tf.group(*[tf.global_variables_initializer(),
                                  tf.tables_initializer()])
    model_spec['variable_init_op'] = variable_init_op
    model_spec["infos"] = inputs['infos'] 
    model_spec["labels"] = inputs['labels'] 
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    #model_spec['auc'] = auc_score 
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
