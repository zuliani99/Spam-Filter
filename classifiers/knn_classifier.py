import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn import neighbors
from sklearn.metrics import accuracy_score
from utlis import print_scores


def callKnn(X, y, X_train, X_test, y_train, y_test):
    print("IMPLEMENTED K-NEAREST NEIGHBOORS")
    scores = cross_val_score(KNearesNeighbour(), X, y, cv = 10)
    print_scores(scores)
    
    print("SKLEARN K-NEAREST NEIGHBOORS")
    kNN = neighbors.KNeighborsClassifier(n_neighbors = 5)
    kNN.fit(X_train, y_train)
    y_pred = kNN.predict(X_test)
    print(f"Accuracy Score of sklearn KNeighborsClassifier in Test Set: {accuracy_score(y_test, y_pred)} \n\n")


class KNearesNeighbour(BaseEstimator):
    def __init__(self):
        self.K = 5
        
        
    def __euclideanDistance(self, vec1, vec2):
        return np.sum(np.sqrt(np.power((vec1 - vec2), 2)))
        
        
    def fit(self, X, Y):
        self.__X_train = X # Private attribute X
        self.__Y_train = Y # Private attribute Y
    
    
    def predict(self, X_test):
        # For each test point i predict the label
        return np.array([self.__predict(test_email) for test_email in X_test])
    
        
    def __predict(self, test_email):
        # Calcualte the distance from the particular point and all the trein points 
        dist = [self.__euclideanDistance(test_email, train_email) for train_email in self.__X_train]
        # Get the first k indices of the sorted vector dist
        k_idx = np.argsort(dist)[: self.K]
        # Return the most popular label
        return Counter(self.__Y_train[k_idx]).most_common(1)[0][0]
    
    
    def accuracy_score(self, y_pred, y_test):
        return np.sum(y_test == y_pred) / len(y_test)
    
    
    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return self.accuracy_score(y_pred, y_test)