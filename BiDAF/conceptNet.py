from BiDAF.hyperparameters import Hyperparams as hp
import sys

class ConceptNet():
    def __init__(self, path=hp.conceptFilter):
        self.data = {}
        self.relation2index = {}
        self.index2relation = {}
        self.relation2index['<NULL>'] = 0
        self.index2relation[0] = '<NULL>'
        index = 1
        for triple in open(path, 'r'):
            triple = triple.replace('\n', '')
            relation, arg1, arg2 = triple.split(' ')
            try:
                a = self.relation2index[relation]
            except KeyError:
                self.relation2index[relation] = index
                self.index2relation[index] = relation
                index += 1
            try:
                self.data[arg1][arg2] = relation
            except KeyError:
                self.data[arg1] = {}
                self.data[arg1][arg2] = relation
            try:
                self.data[arg2][arg1] = relation
            except KeyError:
                self.data[arg2] = {}
                self.data[arg2][arg1] = relation
        print('Loaded concept net.', file=sys.stderr)

    def get_relation(self, word1, word2):
        word1 = '_'.join(word1.lower().split())
        word2 = '_'.join(word2.lower().split())
        if not word1 in self.data:
            return '<NULL>'
        return self.data[word1].get(word2, '<NULL>')

    def c_and_q_relation(self, context, question):
        context = [word.lower() for word in context]
        question = [word.lower() for word in question]
        question = set(question) | set([' '.join(question[i:(i+2)]) for i in range(len(question))])
        ret = ['<NULL>' for _ in context]
        for i in range(len(context)):
            for word in question:
                relation = self.get_relation(context[i], word)
                if relation != '<NULL>':
                    ret[i] = relation
                    break
                relation = self.get_relation(' '.join(context[i:(i+1)]), word)
                if relation != '<NULL>':
                    ret[i] = relation
                    break
                relation = self.get_relation(' '.join(context[i:(i+2)]), word)
                if relation != '<NULL>':
                    ret[i] = relation
                    break
        return ret

net = ConceptNet()
print(net.c_and_q_relation(['fish', 'live', 'in', 'water'], ['salmon', 'live', 'in', 'where']))
