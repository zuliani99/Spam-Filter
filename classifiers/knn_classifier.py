import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utlis import print_confusion_matrix, print_scores


def callKnn(X, y, X_train, X_test, y_train, y_test):
    print("IMPLEMENTED K-NEAREST NEIGHBORS")
    scores = cross_val_score(KNearesNeighbour(), X, y, cv = 10, n_jobs = -1)
    kNN_1 = KNearesNeighbour()
    kNN_1.fit(X_train, y_train)
    y_pred = kNN_1.predict(X_test)
    print_scores(scores, None, accuracy_score(y_test, y_pred))
    
    print("SKLEARN K-NEAREST NEIGHBORS")
    scores = cross_val_score(KNeighborsClassifier(n_neighbors = 5), X, y, cv = 10, n_jobs = -1)
    kNN_2 = KNeighborsClassifier(n_neighbors = 5)
    kNN_2.fit(X_train, y_train)
    y_pred = kNN_2.predict(X_test)
    print_scores(scores, None, accuracy_score(y_test, y_pred))
    print_confusion_matrix(kNN_2, X_test, y_test)


class KNearesNeighbour(BaseEstimator, ClassifierMixin):
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