import numpy as np

def load_csv():
    fread = open("spambase.data", "r")
    data = np.loadtxt(fread, delimiter=",")
    np.random.shuffle(data) # Shuffle dataset
    return data[:,:54], data[:,57]

def print_scores(scores, kernel = None, score_accuracy = None):
    if kernel is not None: print(f"---- {kernel} kernel  ---- ")
    print(f"Cross Val - Min Accuracy: \t\t{str(round(scores.min(), 5))}")
    print(f"Cross Val - Max Accuracy: \t\t{str(round(scores.max(), 5))}")
    print(f"Cross Val - Mean Accuracy: \t\t{str(round(scores.mean(), 5))}")
    print(f"Cross Val - Variance: \t\t\t{str(round(scores.var(), 5))}")
    print(f"Cross Val - Standard Deviation: \t{str(round(scores.std(), 5))}")
    if score_accuracy is not None: print(f"Cross Val - Score Accuracy: \t\t{round(score_accuracy,5)} \n")
    else: print("\n")