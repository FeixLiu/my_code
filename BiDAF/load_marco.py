from BiDAF.hyperparameters import Hyperparams as hp
import json

def load_marco():
    path = hp.marco_dev_data
    with open(path, 'r') as file:
        data = json.load(file)
    print(data['answers']['0'])
    for i in range(len(data['passages']['0'])):
        if data['passages']['0'][i]['is_selected'] == 1:
            print(data['passages']['0'][i])
    print(data['query']['0'])

load_marco()
