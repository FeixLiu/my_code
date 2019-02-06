import tensorflow as tf


def simMat(refc, refq, cLength, qLength, hidden_units):
    """
    :param refc: the reference c
    :param refq: the reference q
    :param cLength: the length of c
    :param qLength: the length of q
    :param hidden_units: the nums of hidden units in the last lstm

    :return simMat: the similarity matrix between c and q
    """
    weights_coMat = tf.Variable(tf.random_normal(dtype=tf.float32, shape=[6 * hidden_units, 1]))
    cExp = tf.tile(tf.expand_dims(refc, 2), [1, 1, qLength, 1])
    qExp = tf.tile(tf.expand_dims(refq, 1), [1, cLength, 1, 1])
    simMat = tf.concat([cExp, qExp, tf.math.multiply(cExp, qExp)], axis=3)
    simMat = tf.reshape(simMat, [-1, 6 * hidden_units])
    simMat = tf.matmul(simMat, weights_coMat)
    simMat = tf.reshape(simMat, [-1, cLength, qLength])
    return simMat


def c2q(refq, qLength, simMat):
    """
    :param refq: the reference q
    :param qLength: the length of q
    :param simMat: the similarity matrix

    :return c2q_attention: the c to q attention
    """
    soft_sim = tf.nn.softmax(simMat, axis=2)
    attention_weight = tf.tile(tf.reduce_sum(soft_sim, axis=2, keepdims=True), [1, 1, qLength])
    c2q_attention = tf.matmul(attention_weight, refq)
    return c2q_attention  # [batch_size, question_length, 2 * lstm_hidden_units]


def q2c(refc, cLength, simMat):
    """
    :param refc: the reference c
    :param cLength: the length of c
    :param simMat: the similarity matrix

    :return q2c_attention: the q to c attention
    """
    soft_sim = tf.nn.softmax(tf.reduce_max(simMat, axis=2), axis=1)
    attented_context_vector = tf.matmul(tf.expand_dims(soft_sim, 1), refc)
    q2c_attention = tf.tile(attented_context_vector, [1, cLength, 1])
    return q2c_attention  # [batch_size, question_length, 2 * lstm_hidden_units]


def calculateG(refc, c2q_attention, q2c_attention):
    """
    :param refc: the reference c
    :param c2q_attention: the c to q attention
    :param q2c_attention: the q to c attention

    :return fuse_vector: the output of the BiDAF
    """
    hu = tf.concat([refc, c2q_attention], axis=2)
    hmu = tf.math.multiply(refc, c2q_attention)
    hmh = tf.math.multiply(refc, q2c_attention)
    fuse_vector = tf.concat([hu, hmu, hmh], axis=2)
    return fuse_vector  # [batch_size, question_length, 8 * lstm_hidden_units]


def biAttention(refc, refq, cLength, qLength, hidden_units):
    """
    :param refc: the first reference c
    :param refq: the second reference q
    :param cLength: the length of c
    :param qLength: the length of q
    :param hidden_units: the nums of hidden units of the last lstm

    :return fuse_vector: the output of the BiDAF
    """
    sim_Mat = simMat(refc, refq, cLength, qLength, hidden_units)
    c2q_attention = c2q(refq, qLength, sim_Mat)
    q2c_attention = q2c(refc, cLength, sim_Mat)
    fuse_vector = calculateG(refc, c2q_attention, q2c_attention)
    return fuse_vector
