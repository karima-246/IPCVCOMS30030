import matplotlib.pyplot as plt
import numpy as np

fpr = np.array([1, 0.0603282, 0.00877532])
tpr = np.array([1, 1, 1])

plt.plot(fpr, tpr, 'o:b', ms = 12)

plt.title("ROC Curve for Classifier Training Stages")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")

plt.show()