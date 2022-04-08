import pandas as pd
import numpy as np
from classifiers.knn_classifier import KNearesNeighbour
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle

def readData():
    data = pd.read_csv("spambase.data", "," ,index_col = False)
    data = shuffle(data)
    return data.iloc[:, :54], data.iloc[:,57]

def callKnn(X_train, X_test, Y_train, Y_test, X, Y):
    knn = KNearesNeighbour()
    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)
    print(f"Accuracy Score: {str(knn.accuracy_score(Y_pred, Y_test))}")

    scores = cross_val_score(knn, X, Y, cv = 10)
    print(f"10 Way Cross Validation Score: {str(np.mean(scores))}")


if __name__ == "__main__":
    X, Y = readData()
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 123)

    #SVM
    #NAIVE BAYES
        
    #KNN
    callKnn(X_train, X_test, Y_train, Y_test, X, Y)