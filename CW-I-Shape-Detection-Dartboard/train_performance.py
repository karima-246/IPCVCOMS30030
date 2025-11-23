import matplotlib.pyplot as plt
import numpy as np

fpr = np.array([1, 0.0603282, 0.00877532])
tpr = np.array([1, 1, 1])

precision = np.array([0.5, 1/1.0603282, 1/1.00877532])
iterations = np.array([0, 1, 2])

plt.plot(iterations, precision, 'o:c', ms = 12)

# plt.title("ROC Curve for Classifier Training Stages")
# plt.xlabel("False Positive Rate (FPR)")
# plt.ylabel("True Positive Rate (TPR)")

plt.title("Precision of Classifier over Training Stages")
plt.xlabel("Training Stage")
plt.ylabel("Precision")

plt.show()