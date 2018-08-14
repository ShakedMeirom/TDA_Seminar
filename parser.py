import os
import pickle

conv_accuracies = []

with open(os.path.join(os.path.dirname(__file__), 'conv_results_run_3.txt'), 'r') as f:
    lines = f.readlines()

for line in lines:
    if "Accuracy" in line:
        conv_accuracies.append(float(line.split(":")[1].split("%")[0])*0.01)

with open(os.path.join(os.path.dirname(__file__), 'result_reddit12K_{}.p'.format('SLayer_Conv')), 'wb') \
        as f:
    pickle.dump(conv_accuracies, f)
