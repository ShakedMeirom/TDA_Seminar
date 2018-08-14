import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open(os.path.join(os.path.dirname(__file__), '../result_high_persistence_test.p'), 'rb') as f:
    accuracies = pickle.load(f)

for key in accuracies:
    plt.plot(accuracies[key], label=str(key))
plt.title("Importance of low persistent points evaluation")
plt.legend(loc='lower right', shadow=True, fancybox=True)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.show()
