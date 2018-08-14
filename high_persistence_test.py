import os
import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import pickle


def _parameters():
    return \
        {
            'data_path': None,
            'epochs': 50,
            'momentum': 0.5,
            'lr_start': 0.1,
            'lr_ep_step': 20,
            'lr_adaption': 0.5,
            'test_ratio': 0.1,
            'batch_size': 128,
            'cuda': False
        }


def high_persistence_test(experiment, provider_path):
    # include only points which have persistence higher than a certain percentage of the mean persistence:
    persistence_thresholds = [1, 0.5, 0]

    accuracies = {}

    for per in persistence_thresholds:
        print('Start run with {} persistence_threshold'.format(per))
        accuracies[per] = experiment(provider_path, _parameters, accuracy_per_epoch=True, persistence_threshold=per,
                                     optimizer=optim.SGD)

    with open(os.path.join(os.path.dirname(__file__), 'result_high_persistence_test.txt'), 'w') as f:
        for key, val in accuracies.items():
            f.write('persistence threshold {}: {}\n'.format(key, val))
        f.write('\n')

    with open(os.path.join(os.path.dirname(__file__), 'result_high_persistence_test.p'), 'wb') as f:
        pickle.dump(accuracies, f)
