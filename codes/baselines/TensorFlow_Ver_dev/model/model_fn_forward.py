"""
Tensorflow graph(node/ops) definitions

"""
import tensorflow as tf
import numpy as np
import model.custom_cell as custom_cell


def fim_cumulative(state, x):
    """ Function for calculating cumulative default probability from FIM
    args:
      state: (tensor), shape = [3, batch], batch of [tau, phi, prob]
      x:     (tensor), shape = [2, batch] => [[f, g], batch]
    return:
      updated state with the same shape
    """
    x = tf.cast(x, tf.float32)
    tau  = state[0, :]
    phi  = state[1, :]
    # P(n) = previous_P + new_P
    prob = state[2, :] + (1 - state[2, :]) * x[0, :]

    new_state = tf.stack([tau, phi, prob])

    return new_state


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

    # Batch_normalization
    if params.batch_norm == "on":
        x_norm = tf.layers.batch_normalization(inputs['features'], training=is_training)
    else:
        x_norm = inputs['features']

    with tf.variable_scope("fim_linear",
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)):

        alpha = tf.layers.dense(inputs=x_norm, units=60,
                #activation=tf.exp,
                activation=tf.nn.sigmoid,
                name='alpha_linear')

        beta = tf.layers.dense(inputs=x_norm, units=60,
                activation=tf.exp,
                name = 'beta_linear')

        logits = tf.stack([alpha, beta], axis=1)

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
    labels = inputs['labels']
    # prepare dummy label for type-2, padd at right side (last node)
    labels = tf.pad(labels, [[0, 0], [0, 1]], constant_values=1)

    # get mask to filter padding event
    pad_event = tf.constant(-1, shape=None)
    not_pad = tf.not_equal(labels, pad_event)
    masks = tf.cast(not_pad, tf.float32)

    # To replace other events as 0
    default_event = tf.constant(1, shape=None)
    is_default = tf.equal(labels, default_event)
    labels = tf.cast(is_default, dtype=tf.float32)

    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)

    with tf.variable_scope("calculate_cumulative"):
        # sub-label for fim model fitting
        sub_labels = inputs['sub_labels']
        sub_not_pad = tf.not_equal(sub_labels, pad_event)
        sub_masks = tf.cast(sub_not_pad, tf.float32)

        sub_is_default = tf.equal(sub_labels, default_event)
        sub_default = tf.cast(sub_is_default, dtype=tf.float32)

        other_event = tf.constant(2, shape=None)
        sub_is_other = tf.equal(sub_labels, other_event)
        sub_other = tf.cast(sub_is_other, dtype=tf.float32)

        alive_event = tf.constant(0, shape=None)
        sub_is_alive = tf.equal(sub_labels, alive_event)
        sub_alive = tf.cast(sub_is_alive, dtype=tf.float32)

        # calculate FIM predictions
        alpha = logits[:, 0, :]
        beta = logits[:, 1, :]
        g = alpha + beta
        g_padded = tf.pad(g, [[0, 0], [1, 0]])
        state = tf.stack([alpha, g_padded[:, :-1]], axis=1)
        state_T = tf.transpose(state)

        eps = 1e-10
        # calculate cumulative default probability by definition
        # make initial state size vary with batch size
        ini_state_T = tf.fill(
            tf.stack(
                [3, tf.shape(inputs['features'])[0]]), 0.0)
        final_state_T = tf.scan(fn=fim_cumulative,
            elems=state_T, initializer=ini_state_T)

        final_state = final_state_T[:, 2, :]
        if params.cum_labels == 6:
            indices = [0, 2, 5, 11, 23, 59]
        else:
            indices = [0, 2, 5, 11, 23, 35, 47, 59]
        agg_state = tf.gather(final_state, indices)
        predictions = tf.transpose(agg_state)
        predictions = tf.clip_by_value(predictions, eps, 1 - eps)

        ## FIM log-likelihood:
        ## alpha terms
        #alpha_alive = sub_alive * tf.log(tf.exp(-alpha/12.0) + eps)
        #alpha_default = sub_default * tf.log(1.0 - tf.exp(-alpha/12.0) + eps)
        #alpha_other = sub_other * tf.log(tf.exp(-alpha/12.0) + eps)
        ## beta terms
        #beta_alive = sub_alive * tf.log(tf.exp(-beta/12.0) + eps)
        #beta_other = sub_other * tf.log(1.0 - tf.exp(-beta/12.0) + eps)

        cross_entropy = sub_default * tf.log(alpha + eps) + (1.0 - sub_default) * tf.log(1.0 - alpha + eps)
        # sum of likelihoods and then reverse the sign
        losses = -(cross_entropy)
        if "indep" not in params.model_version:
            loss = tf.reduce_sum(losses * sub_masks) / tf.reduce_sum(sub_masks)
        else:
            loss_inter = tf.reduce_sum(losses * sub_masks, axis=0) / tf.reduce_sum(sub_masks, axis=0)
            loss = tf.reduce_sum(loss_inter)
    
    # ------------ OPTIMIZER SETTING ----------------------------#
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # Define: training step that minimizes the loss with the Adam optimizer
    if is_training:
        # optimizer = tf.train.AdamOptimizer(params.learning_rate)
        # Wrap adam with weight decay mechanism
        # Because lstm seems to overfitting when window_size increase
        optimizer = tf.contrib.opt.AdamWOptimizer(
                weight_decay=params.weight_decay,
                learning_rate=params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if "indep" not in params.model_version:
            train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_ops = [optimizer.minimize(loss_inter[i], global_step=global_step)
                    for i in range(60)]
            train_op = tf.group(train_ops)
        train_op = tf.group([train_op, update_ops])
    # -----------------------------------------------------------#

    # ------------ METRICS AND SUMMARIES-------------------------#
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        # compute final metrics w.r.t. whole datasets using tf.metrics
        k_aucs = ['auc_{}'.format(i+1) for i in range(params.cum_labels)]
        v_aucs = [tf.metrics.auc(labels=labels[:, i],
                               predictions=predictions[:, i])
                               for i in range(params.cum_labels)]
        metrics = dict(zip(k_aucs, v_aucs))
        #FIXME: average auc over different forward months
        #metrics['auc']  = tf.metrics.mean(tf.reduce_mean(v_aucs))
        metrics['loss'] = tf.metrics.mean(loss)
    # -----------------------------------------------------------


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
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    #model_spec['pred'] = tf.reduce_mean(predictions, axis=0)
    #model_spec['label'] = tf.reduce_mean(labels[:, :-1], axis=0)
    #all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
    #model_spec['num_paras'] = all_trainable_vars


    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
