import tensorflow as tf
from BiDAF.hyperparameters import Hyperparams as hp
import numpy as np
import BiDAF.load_glove as load_glove
import BiDAF.load_squda as load_squda
import BiDAF.bi_attention as bi_attention
import BiDAF.bi_lstm as bi_lstm
import BiDAF.linear_relu as linear_relu
from sklearn.utils import shuffle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

vocab, embd, vocab2index, index2vocab = load_glove.loadGlove()
squda_context, squda_qas, squda_label = load_squda.load_squda(vocab2index)
squda_context, squda_qas, squda_label = shuffle(squda_context, squda_qas, squda_label)
data_size = len(squda_label)
vocab_size = len(vocab)
embedding_dim = hp.embedding_dim_50d
embedding = np.asarray(embd)

# computational graph
embedding_weight = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False)
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = embedding_weight.assign(embedding_placeholder)
keep_prob = tf.placeholder(tf.float32)

with tf.variable_scope('embedding'):
    context_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, hp.context_max_length])  # [None, 64]
    qas_input_ids = tf.placeholder(dtype=tf.int32, shape=[None, hp.qas_max_length])  # [None, 64]
    target = tf.placeholder(dtype=tf.float32, shape=[None, hp.classes])  # [None, 2] model target
    context_embedding = tf.nn.embedding_lookup(embedding_weight, context_input_ids)  # [None, 64, 50] lstm inputs, time steps 64
    qas_embedding = tf.nn.embedding_lookup(embedding_weight, qas_input_ids)  # [None, 32, 50] lstm inputs, time steps 32

with tf.variable_scope('contextLSTM'):
    context_outputs, context_states = bi_lstm.biLSTM(
        inputs=context_embedding,
        input_size=hp.embedding_dim_50d,
        time_steps=hp.context_max_length,  
        hidden_units=hp.lstm_hidden_units,
        batch_size=hp.batch_size,
        project=True
    )
    contextBiLSTM = tf.concat([context_outputs[0], context_outputs[1]], axis=2)  # [8, 64, 2 * 128]
with tf.variable_scope('qasLSTM', reuse=tf.AUTO_REUSE):
    qas_ouputs, qas_states = bi_lstm.biLSTM(
        inputs=qas_embedding,
        input_size=hp.embedding_dim_50d,
        time_steps=hp.qas_max_length,
        hidden_units=hp.lstm_hidden_units,
        batch_size=hp.batch_size,
        project=True
    )
    qasBiLSTM = tf.concat([qas_ouputs[0], qas_ouputs[1]], axis=2)  # [8, 16, 2 * 128]

with tf.variable_scope('biAttention'):
    fuse_vector = bi_attention.biAttention(
        refc=contextBiLSTM,
        refq=qasBiLSTM,
        cLength=hp.context_max_length,
        qLength=hp.qas_max_length,
        hidden_units=hp.lstm_hidden_units
    )  # [8, 64, 8 * 128]

with tf.variable_scope('attentionBiLSTM', reuse=tf.AUTO_REUSE):
    relu_fuse_vector = linear_relu.linearReLU3d(
        inputs=fuse_vector,
        input_length=hp.context_max_length,
        inputs_size=8 * hp.lstm_hidden_units,
        output_size=hp.bidaf_lstm_hidden_units,
        keepProb=None
    )
    bidaf_outputs, bidaf_states = bi_lstm.biLSTM(
        inputs=relu_fuse_vector,
        input_size=8 * hp.lstm_hidden_units,
        time_steps=hp.context_max_length,
        hidden_units=hp.bidaf_lstm_hidden_units,
        batch_size=hp.batch_size,
        project=False
    )
    bidafBiLSTM = tf.concat([bidaf_outputs[0], bidaf_outputs[1]], axis=2)  # [8, 64, 16 * 128]

with tf.variable_scope('selfAttention', reuse=tf.AUTO_REUSE):
    self_fuse_vector = bi_attention.biAttention(
        refc=bidafBiLSTM,
        refq=bidafBiLSTM,
        cLength=hp.context_max_length,
        qLength=hp.context_max_length,
        hidden_units=hp.bidaf_lstm_hidden_units
    )
    relu_self_fuse_vector = linear_relu.linearReLU3d(
        inputs=self_fuse_vector,
        input_length=hp.context_max_length,
        inputs_size=8 * hp.bidaf_lstm_hidden_units,
        output_size=hp.bidaf_lstm_hidden_units
    )

with tf.variable_scope('prediction'):
    relu_sum = tf.add(relu_fuse_vector, relu_self_fuse_vector)
    sum_embedding = tf.reduce_sum(relu_sum, axis=2)
    prediction = linear_relu.linearReLU2d(
        inputs=sum_embedding,
        inputs_size=hp.context_max_length,
        outputs_size=hp.classes,
        keepProb=keep_prob
    )

with tf.variable_scope('costAndTrainOfClassification'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target, logits=prediction))
    train_op = tf.train.AdamOptimizer(hp.learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    for _ in range(hp.epoch):
        start_index = 0
        for i in range(int(data_size / hp.batch_size)):
            xs_context = squda_context[start_index: start_index + hp.batch_size]
            xs_qas = squda_qas[start_index: start_index + hp.batch_size]
            ys = squda_label[start_index: start_index + hp.batch_size]
            start_index += hp.batch_size
            sess.run(train_op, feed_dict={
                context_input_ids: xs_context,
                qas_input_ids: xs_qas,
                target: ys,
                keep_prob: hp.keep_prob
            })
            print(sess.run(cost, feed_dict={
                context_input_ids: xs_context,
                qas_input_ids: xs_qas,
                target: ys,
                keep_prob: hp.keep_prob
            }))
