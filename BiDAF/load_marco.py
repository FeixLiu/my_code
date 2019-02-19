from BiDAF.hyperparameters import Hyperparams as hp
import json
import sys
import nltk


def analyseContext(context):
    """
    :param context: the all paragraphs of the passage

    :return positive: positive paragraphs with the question
    :return negative: negative paragraphs with the question
    """
    positive = []
    negative = []
    for i in range(len(context)):
        if context[i]['is_selected'] == 1:
            positive.append(context[i]['passage_text'])
        if context[i]['is_selected'] == 0:
            negative.append(context[i]['passage_text'])
    return positive, negative


def convert2index(data, vocab2index):
    """
    :param data: the data need to convert to index from word
    :param vocab2index: the vocabulary to index dictionary

    :return data_list: the list of the index of the input data
    """
    tokens = nltk.word_tokenize(data)
    data_list = []
    for word in tokens:
        word = word.lower()
        try:
            index = vocab2index[word]
        except KeyError:
            index = 0
        data_list.append(index)
    if len(data_list) < hp.context_max_length:
        for _ in range(hp.context_max_length - len(data_list)):
            data_list.append(0)
    return data_list


def loadMarco(vocab2index):
    """
    :param vocab2index: the vocabulary to index dictionary

    :return marco_contexts: the paragraphs
    :return marco_queries: the questions
    :return marco_labels: the labels whether the question is relating with the paragraph or not
    :return marco_answers: the answers, if the question is not relating with the paragraph or the question has no answer
                        , then the answer is [0 ...]
    """
    path = hp.marco_dev_data
    with open(path, 'r') as file:
        data = json.load(file)
    marco_contexts = []
    marco_queries = []
    marco_labels = []
    marco_answers = []
    for i in range(len(data['answers'])):
        i = str(i)
        answer = data['answers'][i]
        context = data['passages'][i]
        query = data['query'][i]
        query = convert2index(query, vocab2index)[:hp.qas_max_length]
        positive, negative = analyseContext(context)
        for i in range(len(positive)):
            marco_contexts.append(convert2index(positive[i], vocab2index))
            marco_queries.append(query)
            marco_labels.append([0., 1.])
            for j in range(len(answer)):
                marco_answers.append(convert2index(answer[j], vocab2index)[:hp.answer_max_length])
                if j >= 1:
                    marco_contexts.append(convert2index(positive[i], vocab2index))
                    marco_queries.append(query)
                    marco_labels.append([0., 1.])
        for i in range(len(negative)):
            marco_contexts.append(convert2index(negative[i], vocab2index))
            marco_queries.append(query)
            marco_answers.append([0 for _ in range(hp.answer_max_length)])
            marco_labels.append([1., 0.])
    print('Loaded Marco.', file=sys.stderr)
    return marco_contexts, marco_queries, marco_labels, marco_answers
