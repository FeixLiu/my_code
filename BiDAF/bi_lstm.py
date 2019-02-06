import tensorflow as tf


def biLSTM(inputs, input_size, time_steps, hidden_units, batch_size, project=False):
    """
    :param inputs: the input tensor shape is [batch size, time steps, inputs size]
    :param input_size: the 3rd dimension of the inputs
    :param time_steps: the 2ne dimension of the inputs
    :param hidden_units: the units of the lstm
    :param batch_size: the 1st dimension of the inputs
    :param project: whether do the wx_plus_b or not

    :return outputs: the all hidden states of the lstm
    :return states: the final states of the lstm
    """
    if project:
        weights = tf.Variable(tf.random_normal(shape=[input_size, hidden_units]))
        biases = tf.Variable(tf.constant(0.1, shape=[hidden_units]))
        x = tf.reshape(inputs, [-1, input_size])
        x_in = tf.add(tf.matmul(x, weights), biases)
        x_in = tf.reshape(x_in, [-1, time_steps, hidden_units])
    else:
        x_in = inputs
    cell_fw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units, forget_bias=1.0, state_is_tuple=True)
    cell_bw = tf.nn.rnn_cell.LSTMCell(num_units=hidden_units, forget_bias=1.0, state_is_tuple=True)
    initial_state_fw = cell_fw.zero_state(batch_size, dtype=tf.float32)
    initial_state_bw = cell_bw.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=cell_fw,
        cell_bw=cell_bw,
        inputs=x_in,
        initial_state_fw=initial_state_fw,
        initial_state_bw=initial_state_bw
    )
    return outputs, states
