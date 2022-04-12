import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator


class KNearesNeighbour(BaseEstimator):
    def __init__(self):
        #print("KNN Classifier Started")
        self.K = 5
        
        
    def __euclideanDistance(self, vec1, vec2):
        return np.sum(np.sqrt((vec1 - vec2) ** 2))
        
        
    def fit(self, X, Y):
        self.__X_train = X # Private attribute X
        self.__Y_train = Y # Private attribute Y
    
    
    def predict(self, X_test):
        y_pred = []
        #print("Predicting Values...")
        for test_p in X_test:
            dist = [self.__euclideanDistance(test_p, train_p) for train_p in self.__X_train]
            dist = pd.Series(dist)
            sort_dist_k = dist.sort_values()[:self.K]
            ham_spam_count = Counter(self.__Y_train[sort_dist_k.index])
            y_pred.append(ham_spam_count.most_common()[0][0])

        return y_pred
    
    
    def accuracy_score(self, y_pred, y_test):
        return np.sum(y_test == y_pred) / len(y_test)
    
    
    def score(self, x_test, y_test):
        y_pred = self.predict(x_test)
        return self.accuracy_score(y_pred, y_test)
    
