class Hyperparams:
    glove_data = '../glove/glove.840B.300d.txt'
    glove_data_50d = '../glove/glove.6B.50d.txt'
    marco_train_data = '../marco/train_v2.1.json'
    marco_dev_data = '../marco/dev_v2.1.json'
    squda_train_data = '../squda/train-v2.0.json'
    squda_dev_data = '../squda/dev-v2.0.json'
    embedding_dim_50d = 50
    embedding_dim = 300
    batch_size = 8
    context_max_length = 64
    qas_max_length = 16
    epoch = 10
    lstm_hidden_units = 128
    bidaf_lstm_hidden_units = 1024
    learning_rate = 0.001
    keep_prob = 0.5
    classes = 2
