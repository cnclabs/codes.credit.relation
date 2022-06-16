"""
Tensorflow graph(node/ops) definitions

"""
import tensorflow as tf
import numpy as np
import model.custom_cell as custom_cell


def get_length(sequence):
    """Get sequence length
    Assume sequence has shape: batch * window_size * feature_size
    and the padding value is 0
    """
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


def fim_cumulative(state, x):
    """ Function for calculating cumulative default probability from FIM
    args:
      state: (tensor), shape = [3, batch], batch of [tau, phi, prob]
      x:     (tensor), shape = [2, batch] => [[f, g], batch]
    return:
      updated state with the same shape
    """
    x = tf.cast(x, tf.float32)
    ## tau(n) = previous_tau + 1
    #tau  = state[0, :] + 1.0
    ## Phi(n) = previous_Phi + new_Phi
    #phi  = state[1, :] * (tau - 1.0) + x[1, :]
    ## P(n) = previous_P + new_P
    #prob = state[2, :] + tf.exp(-phi/12.0) * (1.0 - tf.exp(-x[0, :]/12.0))
    #new_state = tf.stack([tau, phi, prob])
    prob = state[2, :] +  (1.0 - state[0, :] - state[1, :]) * x[0, :]
    prev_default = state[0, :] + x[0, :]
    prev_other = state[1, :] + x[1, :]
    new_state = tf.stack([prev_default, prev_other, prob])

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

    if 'fim' not in params.model_version:
        sentence = tf.reshape(x_norm, [-1, params.window_size, params.feature_size])
        # ------ Recurrent cell -------- #
        length = get_length(sentence)

        # Use orthorgonal initializer to initialize LSTM cell
        with tf.variable_scope("cell", initializer=tf.orthogonal_initializer()):

            keep_prob = (1.0 - params.dropout_rate) if is_training else 1.0

            if 'mlp' in params.model_version:
                logits = tf.layers.dense(inputs=x_norm, units=params.layer1_num_units, activation=tf.nn.sigmoid)
                logits = tf.layers.dense(inputs=logits, units=params.layer2_num_units, activation=tf.nn.sigmoid)
            else:
                if 'lstm' in params.model_version:
                    cell = tf.nn.rnn_cell.LSTMCell(params.lstm_num_units)
                elif 'lstm_layer_norm' in params.model_version:
                    layer_norm_switch = ( params.layer_norm == "on" )
                    cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
                            params.lstm_num_units,
                            layer_norm=layer_norm_switch,
                            dropout_keep_prob=keep_prob)
                else:
                    cell = tf.nn.rnn_cell.GRUCell(params.lstm_num_units)

                if 'lstm_layer_norm' not in params.model_version:
                    dropout_cell = tf.nn.rnn_cell.DropoutWrapper(cell,
                            input_keep_prob=1.0, output_keep_prob=keep_prob)
                else:
                    dropout_cell = cell

                output, final_state = tf.nn.dynamic_rnn(
                    cell=dropout_cell,
                    inputs=sentence,
                    dtype=tf.float32,
                    sequence_length=length)

        # ------ Output cell -------- #
        # Type-2 network, add extra nodes at last, but not using it
        with tf.variable_scope("type2"):
            if 'lstm' in params.model_version:
                if hasattr(params, 'hidden'):
                    logits = tf.layers.dense(inputs=final_state.h, units=params.hidden)
                    logits = tf.layers.dense(inputs=logits, units=params.cum_labels + 1)
                else:
                    logits = tf.layers.dense(inputs=final_state.h, units=params.cum_labels + 1)

            elif 'gru' in params.model_version:
                logits = tf.layers.dense(inputs=final_state, units=params.cum_labels + 1)
            else:
                logits = tf.layers.dense(inputs=logits, units=params.cum_labels + 1)

            if 'type2' in params.model_version:
                logits = tf.nn.softmax(logits, axis=1)
                logits = tf.cumsum(logits, axis=1)
            else:
                logits = tf.nn.sigmoid(logits)

            eps = 5e-8  #FIXME: warning, cannot set as 1e-8, will cause loss to be nan
            logits = tf.clip_by_value(logits, eps, 1 - eps)
    else:
        with tf.variable_scope("fim",
                initializer=tf.keras.initializers.glorot_normal(seed=124)):
                #initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=None)):

            def poisson_prob(x, period=12.0):
                return (1.0 - tf.exp(-x / period))

            if 'mlp' in params.model_version:
            #    x_norm = tf.layers.dense(inputs=x_norm, units=params.layer1_num_units, activation=tf.nn.sigmoid)
                x_norm = tf.layers.dense(inputs=x_norm, units=64, activation=tf.nn.sigmoid)

            alpha = tf.layers.dense(inputs=x_norm, units=60,
                    activation=tf.exp,
                    name='alpha_linear')

            beta = tf.layers.dense(inputs=x_norm, units=60,
                    activation=tf.exp,
                    name = 'beta_linear')

            logits = tf.stack([poisson_prob(alpha), poisson_prob(beta)], axis=1)

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

    if 'fim' in params.model_version:
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
            #g = alpha + beta
            #g_padded = tf.pad(g, [[0, 0], [1, 0]])
            #state = tf.stack([alpha, g_padded[:, :-1]], axis=1)
            state = tf.stack([alpha, beta], axis=1)
            state_T = tf.transpose(state)

            eps = 1e-8
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

            # FIM log-likelihood:
            # alpha terms
            alpha_alive = sub_alive * tf.log(1.0 - alpha + eps)
            alpha_default = sub_default * tf.log(alpha + eps)
            alpha_other = sub_other * tf.log(1.0 - alpha + eps)
            # beta terms
            beta_alive = sub_alive * tf.log(1.0 - beta + eps)
            beta_other = sub_other * tf.log(beta + eps)
            # sum of likelihoods and then reverse the sign
            losses = -(alpha_alive + alpha_default + alpha_other + beta_alive + beta_other)
            loss = tf.reduce_sum(losses * sub_masks) / tf.reduce_sum(sub_masks)

    else:
        predictions = logits
        # ------------ LOSS FUNCTIONS -------------------------------#
        # Define loss and accuracy (we need to apply a mask to account for padding)
        # cross-entropy
        losses = (-1) * ( labels * tf.log(logits) + (1 - labels) * tf.log(1 - logits) )
        ## NLL loss
        #losses = (-1) * labels * tf.log(logits)
        ##PoissonNLL
        #losses = logits - labels * tf.log(logits)
        ## Aggregated_default_loss
        #loss_agg = tf.losses.mean_squared_error(tf.reduce_mean(labels), tf.reduce_mean(logits))
        # -----------------------------------------------------------#
        loss = tf.reduce_mean(losses * masks)

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
        #optimizer = tf.train.MomentumOptimizer(
        #        learning_rate=params.learning_rate,
        #        momentum=0.9
        #        )
        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss, global_step=global_step)
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
    all_trainable_vars = tf.reduce_sum([tf.reduce_prod(v.shape) for v in tf.trainable_variables()])
    model_spec['num_paras'] = all_trainable_vars


    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
