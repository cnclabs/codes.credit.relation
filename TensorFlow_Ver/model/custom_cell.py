import tensorflow as tf
from tensorflow.keras.constraints import NonNeg
from tensorflow.python.ops import init_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class CustomLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 state_is_tuple=True,
                 activation=None,
                 reuse=None,
                 name=None,
                 dtype=None):
        super(CustomLSTMCell, self).__init__(num_units,
                                             forget_bias,
                                             state_is_tuple,
                                             activation,
                                             reuse,
                                             name,
                                             dtype)
    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth, 4 * self._num_units],
            constraint=NonNeg())
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[4 * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype),
            constraint=NonNeg())

        self.built = True