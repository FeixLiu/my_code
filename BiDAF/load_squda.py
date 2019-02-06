from BiDAF.hyperparameters import Hyperparams as hp
import numpy as np
import json
import sys


def load_squda(vocab2index):
    """
    :param vocab2index: the vocabulary to index dictionary

    :return np.array(squda_context): the matrix of all context
    :return np.array(squda_qas): the matrix of all questions
    :return np.array(squda_label): the matrix of all label
    """
    filepath = hp.squda_data
    squda_data = []
    with open(filepath, 'r') as file:
        data = json.load(file)
        data = data['data']
        for i in range(len(data)):
            paragraph = {}
            for j in range(len(data[i]['paragraphs'])):
                context = data[i]['paragraphs'][j]['context']
                context_list = convert2list(context, vocab2index)
                paragraph['context'] = context_list[:hp.context_max_length]
                pos_qas = []
                for k in range(len(data[i]['paragraphs'][j]['qas'])):
                    question = data[i]['paragraphs'][j]['qas'][k]['question']
                    question_list = convert2list(question, vocab2index)
                    pos_qas.append(question_list[:hp.qas_max_length])
                paragraph['pos'] = pos_qas
            squda_data.append(paragraph)
    squda_data = load_negative(squda_data)
    squda_context, squda_qas, squda_label = convert2tensor(squda_data)
    print('Loaded Squda.', file=sys.stderr)
    return np.array(squda_context), np.array(squda_qas), np.array(squda_label)


def convert2tensor(squda_data):
    """
    :param squda_data: the data dictionary

    :return squda_context: the matrix of context
    :return squda_qas: the matrix of question
    :return squda_label: the matrix of label
    """
    squda_context = []
    squda_qas = []
    squda_label = []
    for i in range(len(squda_data)):
        context = squda_data[i]['context']
        for j in range(len(squda_data[i]['pos'])):
            squda_context.append(context)
            squda_qas.append(squda_data[i]['pos'][j])
            squda_label.append([0., 1.])
        for j in range(len(squda_data[i]['neg'])):
            squda_context.append(context)
            squda_qas.append(squda_data[i]['neg'][j])
            squda_label.append([1., 0.])
    return squda_context, squda_qas, squda_label


def convert2list(data, vocab2index):
    """
    :param data: the data need to convert to index from word
    :param vocab2index: the vocabulary to index dictionary

    :return data_list: the list of the index of the input data
    """
    data = data.split(' ')
    data_list = []
    for word in data:
        word = word.lower()
        punctuation = []
        while (word[-1:] > 'z' or (word[-1:] < 'a' and word[-1:] > '9') or word[-1:] < '0') and word[-1:] <= '~' and word:
            punctuation.append(word[-1:])
            word = word[:-1]
        try:
            index = vocab2index[word]
        except KeyError:
            index = 0
        data_list.append(index)
        punctuation.reverse()
        if punctuation:
            for i in punctuation:
                try:
                    index = vocab2index[i]
                except KeyError:
                    index = 0
                data_list.append(index)
    if len(data_list) < hp.context_max_length:
        for _ in range(hp.context_max_length - len(data_list)):
            data_list.append(0)
    return data_list

def load_negative(squda_data):
    """
    :param squda_data: the dictionary of squda data

    :return squda_data: the dictionary of squda data with negative questions
    """
    total = len(squda_data)
    for i in range(len(squda_data)):
        paragraph = squda_data[i]
        neg_pos = []
        for _ in range(len(squda_data[i]['pos'])):
            a = np.random.randint(0, total)
            while a == i:
                a = np.random.randint(0, total, 1)[0]
            b = np.random.randint(0, len(squda_data[a]['pos']))
            neg_pos.append(squda_data[a]['pos'][b])
        paragraph['neg'] = neg_pos
    return squda_data
