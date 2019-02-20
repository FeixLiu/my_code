from BiDAF.hyperparameters import Hyperparams as hp
from BiDAF.load_glove import loadGlove
import json
import sys


def load_conceptNet(vocab2index):
    """
    :param vocab: the vocabulary of glove
    """
    index = 0.0
    def _get_lan_and_w(arg):
        arg = arg.strip('/').split('/')
        return arg[1], arg[2]
    writer = open(hp.conceptFilter, 'w', encoding='utf-8')
    with open(hp.conceptNet, 'r') as file:
        for line in file:
            index += 1.0
            if index % 10000 == 0:
                print(index / 32755210, file=sys.stderr)
            fs = line.split('\t')
            relation, arg1, arg2 = fs[1].split('/')[-1], fs[2], fs[3]
            lan1, w1 = _get_lan_and_w(arg1)
            flag = 1
            if lan1 != 'en':
                continue
            lan2, w2 = _get_lan_and_w(arg2)
            if lan2 != 'en':
                continue
            obj = json.loads(fs[-1])
            if obj['weight'] < 1.0:
                continue
            for w in w1.split('_'):
                try:
                    vocab2index[w]
                except KeyError:
                    flag = 0
                    break
            for w in w2.split('_'):
                try:
                    vocab2index[w]
                except KeyError:
                    flag = 0
                    break
            if flag:
                writer.write('%s %s %s\n' % (relation, w1, w2))
    writer.close()


vocab, embd, vocab2index, index2vocab = loadGlove()
load_conceptNet(vocab2index)
