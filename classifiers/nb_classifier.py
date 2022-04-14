import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from utlis import print_scores
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


def callNb(X, y, X_train, X_test, y_train, y_test):
    scores = cross_val_score(NaiveBayes(), X, y, cv = 10)
    print("IMPLEMENTED NAIVE BAYES")
    print_scores(scores)

    print("SKLEARN GAUSSIAN NB")
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print(f"Accuracy Score of sklearn GaussianNB in Test Set: {accuracy_score(y_test, y_pred)} \n\n")
    

class NaiveBayes(BaseEstimator):
    def fit(self, X, y):
        n_samples, n_features = X.shape # Get the dataset shape
        self.__classes = np.unique(y) # Get the unique classes
        n_classes = len(self.__classes)

        # Initialize useful np.array
        self.__mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.__var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.__priors = np.zeros(n_classes, dtype=np.float64)

        # Fill the np.array
        for c in self.__classes:
            X_c = X[y == c] # Get all email with labeled with c 
            
            # Fill the row int(c) of mean vector with the mean of all the feature
            self.__mean[int(c), :] = X_c.mean(axis = 0) 
            # Fill the row int(c) of variance vector with the varaince of all the feature
            self.__var[int(c), :] = X_c.var(axis = 0) + 1e-128 
            # Fill the row int(c) of priors probability vector with its prior prabability
            self.__priors[int(c)] = float(X_c.shape[0] / n_samples)


    def predic(self, X_test):
        return np.array([self.__predict(test_email) for test_email in X_test])


    def __predict(self, x):
        post_probs = []

        # Calculate posterior probability for each class 
        for c in self.__classes:
            prior_prob = np.log(self.__priors[int(c)])
            post_prob = np.sum(np.log(self.__gaussianDistribution(int(c), x) + 1e-128))
            post_probs.append((prior_prob + post_prob))

        # Return class with highest posterior probability
        return self.__classes[np.argmax(post_probs)]


    # Gaussian Distribution Function
    def __gaussianDistribution(self, class_idx, x):
        num = np.exp(-(np.power((x - self.__mean[class_idx]), 2)) / (2 * self.__var[class_idx]))
        den = np.sqrt(2 * np.pi * self.__var[class_idx])
        return num / den


    def accuracy_score(self, y_pred, y_test):
        return np.sum(y_test == y_pred) / len(y_test)


    def score(self, x_test, y_test):
        y_pred = self.predic(x_test)
        return self.accuracy_score(y_test, y_pred)