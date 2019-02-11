from ptr_generator.hyperparameters import Hyperparameters as hp
import tensorflow as tf
from ptr_generator.load_glove import loadGlove
from ptr_generator.load_marco import loadMarco
from sklearn.utils import shuffle
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

vocab, embd, vocab2index, index2vocab = loadGlove()
marco_contexts, marco_queries, marco_labels, marco_answers = loadMarco(vocab2index)
marco_contexts, marco_queries, marco_labels, marco_answers = shuffle(marco_contexts, marco_queries, marco_labels, marco_answers)
data_size = len(marco_labels)
vocab_size = len(vocab)
embedding_dim = hp.embedding_dim_50d
embedding = np.asarray(embd)

# computational graph
embedding_weight = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False)
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = embedding_weight.assign(embedding_placeholder)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
