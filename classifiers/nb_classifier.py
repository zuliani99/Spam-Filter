import numpy as np
from sklearn.base import BaseEstimator

class NaiveBayes(BaseEstimator):
    def score(self, X, Y):
        px_spam_i = ((2 * np.pi * self.o2_spam) ** (-1. / 2)) * np.exp(-1. / (2 * self.o2_spam) * np.power(X - self.mu_spam, 2))
        px_ham_i = ((2 * np.pi * self.o2_ham) ** (-1. / 2)) * np.exp(-1. / (2 * self.o2_ham) * np.power(X - self.mu_ham, 2))
        
        px_spam = np.prod(px_spam_i, axis = 1)
        px_ham = np.prod(px_ham_i, axis = 1)
        
        p_spam = px_spam * self.prob_spam
        p_ham = px_ham * self.prob_ham
                           
        y_pred = np.argmax([p_ham, p_spam], axis = 0)
        return np.mean(y_pred == Y)

    def fit(self, X, Y):
        self.spam = X[Y == 1, :54]
        self.ham = X[Y == 0, :54]
        
        self.n_spam = self.spam.shape[0] 
        self.n_ham = self.ham.shape[0] 
        self.N = float(self.n_spam + self.n_ham)

        self.prob_spam = self.n_spam / self.N
        self.prob_ham = self.n_ham / self.N
        
        self.mu_spam = np.mean(self.spam, axis = 0)
        self.mu_ham = np.mean(self.ham, axis = 0)
        
        self.o2_spam = np.var(self.spam, axis = 0) + 1e-128
        self.o2_ham = np.var(self.ham, axis = 0) + 1e-128