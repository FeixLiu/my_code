import tensorflow as tf


def linearReLU3d(inputs, input_length, inputs_size, output_size, keepProb=None):
    """
    :param inputs: the input vector [batch size, input length, input size]
    :param input_length: the 2nd dimension of the inputs
    :param inputs_size: the 3rd dimension of the inputs
    :param output_size: the output size of the linear relu
    :param keepProb: whether do the dropout

    :return relu: the tensor after relu(wx_plus_b)
    """
    inputs = tf.reshape(inputs, [-1, inputs_size])
    weights = tf.Variable(tf.random_normal(shape=[inputs_size, output_size]))
    biases = tf.Variable(tf.constant(0.1, shape=[output_size]))
    wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
    relu = tf.nn.relu(wx_plus_b)
    if keepProb is not None:
        relu = tf.nn.dropout(relu, keepProb)
    relu = tf.reshape(relu, [-1, input_length, output_size])
    return relu


def linearReLU2d(inputs, inputs_size, outputs_size, keepProb=None):
    """
    :param inputs: the input vector [batch size, input size]
    :param inputs_size: the 2rd dimension of the inputs
    :param output_size: the output size of the linear relu
    :param keepProb: whether do the dropout

    :return relu: the tensor after relu(wx_plus_b)
    """
    weights = tf.Variable(tf.random_normal(shape=[inputs_size, outputs_size]))
    biases = tf.Variable(tf.constant(0.1, shape=[outputs_size]))
    wx_plus_b = tf.add(tf.matmul(inputs, weights), biases)
    relu = tf.nn.relu(wx_plus_b)
    if keepProb is not None:
        relu = tf.nn.dropout(relu, keepProb)
    return relu
