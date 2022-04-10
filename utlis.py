import numpy as np

def load_csv():
    fread = open("files/spambase.data", "r")
    data = np.loadtxt(fread, delimiter=",")
    np.random.shuffle(data) # Shuffle dataset
    return data[:,:54], data[:,57]