from BiDAF.hyperparameters import Hyperparams as hp
import csv


def load_conceptNet():
    file = open(hp.conceptNet, 'r')
    reader = csv.reader(file)
    print(reader)


load_conceptNet()
