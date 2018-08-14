import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

Layers = ["SLayer_Conv", "SLayer_Weighted_Avg", "SLayer"]
accuracies = {}

for layer in Layers:
    with open(os.path.join(os.path.dirname(__file__), '../result_reddit12K_{}.p'.format(layer)), 'rb') as f:
        accuracies[layer] = pickle.load(f)
        accuracies[layer] = np.squeeze(accuracies[layer])

for key in accuracies:
    plt.plot(accuracies[key], label=key)
plt.title("Evaluation of different designated layers formulations")
plt.legend(loc='lower right', shadow=True, fancybox=True)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.show()

ticks = np.linspace(0, 0.5, num=21)
plt.bar(accuracies.keys(), [max(accuracies[key]) for key in accuracies], zorder=3, color=['blue','orange','green'])
plt.title("Max Accuracy of different designated layers formulations")
plt.ylabel("Test Accuracy")
plt.yticks(ticks)
plt.grid(zorder=0)
plt.show()

print([max(accuracies[key]) for key in accuracies])
