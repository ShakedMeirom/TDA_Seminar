import os
import numpy as np
from torch import optim
from chofer_torchex.nn import SLayer, SLayer_Conv, SLayer_Weighted_Avg
import pickle


def _parameters():
    return \
        {
            'data_path': None,
            'epochs': 300,
            'momentum': 0.5,
            'lr_start': 0.1,
            'lr_ep_step': 20,
            'lr_adaption': 0.5,
            'test_ratio': 0.1,
            'batch_size': 128,
            'cuda': False
        }


def accuracy_test(experiment, provider_path):
    SLayers_dict = {"SLayer_Weighted_Avg": SLayer_Weighted_Avg, "SLayer_Conv": SLayer_Conv, "SLayer": SLayer}

    for layer_ver in SLayers_dict:
        accuracies = []
        n_runs = 1
        for i in range(1, n_runs + 1):
            print('Start run {}'.format(i))
            result = experiment(provider_path, _parameters, accuracy_per_epoch=True, slayer=SLayers_dict[layer_ver],
                                optimizer=optim.SGD)
            accuracies.append(result)

        with open(os.path.join(os.path.dirname(__file__), 'result_reddit12K_{}.txt'.format(layer_ver)), 'w')\
                as f:
            for i, r in enumerate(accuracies):
                f.write('Run {}: {}\n'.format(i, r))
            f.write('\n')
            f.write('mean: {}\n'.format(np.mean(accuracies)))

        with open(os.path.join(os.path.dirname(__file__), 'result_reddit12K_{}.p'.format(layer_ver)), 'wb') \
                as f:
            pickle.dump(accuracies, f)
