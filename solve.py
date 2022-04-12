import pandas as pd
import numpy as np
from classifiers.svm_classifier import Svm 

def readData():
    data = pd.read_csv("files/spambase.data", "," ,index_col = False)
    return data.iloc[:, :54], data.iloc[:,57]

if __name__ == "__main__":
    X,Y = readData()
    #SVM
    svm = Svm(X,Y)
    svm.result()
    #NAIVE BAYES
    #KNN