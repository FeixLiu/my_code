import sys
from ptr_generator.hyperparameters import Hyperparameters as hp

def loadGlove():
    """
    :return vocab: all vocabulary of the glove
    :return embd: the embedding matrix
    :return vocab2index: vocabulary to index dictionary
    :return index2vocab: index to vocabulary dictionary
    """
    filepath = hp.glove_data_50d
    vocab = []
    embd = []
    vocab2index = {}
    index2vocab = {}
    index = 0
    vocab.append('unk')
    embd.append([0] * hp.embedding_dim_50d)
    vocab2index['unk'] = 0
    index2vocab[0] = 'unk'
    with open(filepath, 'r') as file:
        for line in file:
            row = line.strip().split(' ')
            index += 1
            vocab2index[row[0]] = index
            index2vocab[index] = row[0]
            vocab.append(row[0])
            embd.append(row[1:])
    print('Loaded Glove.', file=sys.stderr)
    return vocab, embd, vocab2index, index2vocab



