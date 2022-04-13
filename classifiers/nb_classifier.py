import numpy as np
from sklearn.base import BaseEstimator

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
            X_c = X[y == c]
            self.__mean[int(c), :] = X_c.mean(axis=0)
            self.__var[int(c), :] = X_c.var(axis=0) + 1e-128
            self.__priors[int(c)] = float(X_c.shape[0] / n_samples)


    def predic(self, X):
        return np.array([self.__predict(x) for x in X])


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
        num = np.exp(-((x - self.__mean[class_idx]) ** 2) / (2 * self.__var[class_idx]))
        den = np.sqrt(2 * np.pi * self.__var[class_idx])
        return num / den


    def score(self, x_test, y_test):
        y_pred = self.predic(x_test)
        return np.sum(y_test == y_pred) / len(y_test)
    
    def accuracy_score(self, y_pred, y_test):
        return np.sum(y_test == y_pred) / len(y_test)