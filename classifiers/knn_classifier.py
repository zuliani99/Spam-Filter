import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import BaseEstimator

class KNearesNeighbour(BaseEstimator):
    def __init__(self):
        print("KNN Classifier Started")
        self.K = 5
        
    def __euclideanDistance(self, vec1, vec2):
        return np.sum(np.sqrt((vec1 - vec2)**2))
        
    def fit(self, X, Y):
        self.__X_train = X # Private attribute X
        self.__Y_train = Y # Private attribute Y
    
    def predict(self, X_test):
        y_pred = []
        print("Predicting Values...")
        for test_p in X_test.to_numpy():
            dist = [self.__euclideanDistance(test_p, train_p) for train_p in self.__X_train.to_numpy()]
            df_dist = pd.DataFrame(data = dist, columns = ['dist'], index = self.__Y_train.index)
            df_sorted_k = df_dist.sort_values(by = ['dist'])[:self.K]
            ham_spam_count = Counter(self.__Y_train[df_sorted_k.index])
            y_pred.append(ham_spam_count.most_common()[0][0])
        return y_pred
    
    def accuracy_score(self, Y_pred, Y_test):
        comparison = Y_pred == Y_test
        return len(comparison[comparison]) / len(comparison)

    def score(self, X, Y):
        pred = self.predict(X)
        comparison = pred == Y
        return len(comparison[comparison]) / len(comparison)