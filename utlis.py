import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def load_csv():
    fread = open("spambase.data", "r")
    data = np.loadtxt(fread, delimiter=",")
    np.random.shuffle(data) # Shuffle dataset
    return data[:,:54], data[:,57]

def print_scores(scores, kernel = None, test_accuracy = None):
    if kernel is not None: print(f"---- {kernel} kernel  ---- ")
    print(f"Cross Val - Min Accuracy: \t\t{str(round(scores.min(), 5))}")
    print(f"Cross Val - Mean Accuracy: \t\t{str(round(scores.mean(), 5))}")
    print(f"Cross Val - Max Accuracy: \t\t{str(round(scores.max(), 5))}")
    print(f"Cross Val - Variance: \t\t\t{str(round(scores.var(), 5))}")
    print(f"Cross Val - Standard Deviation: \t{str(round(scores.std(), 5))}")
    if test_accuracy is not None: print(f"Test Set - Accuracy Score: \t\t{round(test_accuracy, 5)} \n")
    
def print_confusion_matrix(classifier, x_test, y_test, p = None, k = None):
    cm = ConfusionMatrixDisplay.from_estimator(classifier, x_test, y_test)
    if p is not None: cm.ax_.set_title(f'{p} {k} kernel')
    plt.show()