import numpy as np
from classifiers.knn_classifier import KNearesNeighbour
from classifiers.nb_classifier import NaiveBayes
from utlis import load_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def callNb(X, y):
    scores = cross_val_score(NaiveBayes(), X, y, cv = 10)
    print("NAIVE BAYES")
    print(f"Cross Val - Min Accuracy: \t{str(round(scores.min(), 5))}")
    print(f"Cross Val - Mean Accuracy: \t{str(round(scores.mean(), 5))}")
    print(f"Cross Val - Max Accuracy: \t{str(round(scores.max(), 5))}" + "\n")


def callKnn(X_train, y_train, X_test, y_test, X, y):
    print("K NEAREST NEIGHBOORS")
    
    knn = KNearesNeighbour()
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    print(f"Train Test Split - Accuracy Score: \t{str(knn.accuracy_score(y_pred, y_test))}")

    scores = cross_val_score(knn, X, y, cv = 10)
    print(f"Cross Val - Min Accuracy: \t{str(round(scores.min(), 5))}")
    print(f"Cross Val - Mean Accuracy: \t{str(round(scores.mean(), 5))}")
    print(f"Cross Val - Max Accuracy: \t{str(round(scores.max(), 5))}")


def start():
    X, y = load_csv()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    #SVM
    
    #NAIVE BAYES
    callNb(X, y)
        
    #KNN
    callKnn(X_train, y_train, X_test, y_test, X, y)
    
    
if __name__ == "__main__":
   start()