import copy
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from utlis import print_scores, print_confusion_matrix

CV = 10
KERNELS = ["linear", "poly", "rbf"]
PRINT = ["Measures of kernels without angular kernel evaluation:",
          "Measures of kernels with angular kernel evaluation:"]


class Svm:
    def __init__ (self, X, y):
        self.X = self.tf_idf(X)
        self.norm_X = self.normalization(self.X)
        self.y = y
        self.non_normalized_data = train_test_split(self.X, self.y, test_size = 0.3)
        self.normalized_data = train_test_split(self.norm_X , self.y, test_size = 0.3)


    # Calculate the term frequency of each word in each mail
    def tf_idf(self, X):
        idf = np.log10(X.shape[0] / (X != 0).sum(0))
        return X / 100.0 * idf


    # Normalize dataset
    def normalization(self,X):
        x = copy.deepcopy(X)
        norms = preprocessing.normalize(x)
        nonzero = norms > 0
        x[nonzero] /= norms[nonzero]
        return x


	# Create the specified classifier
    def create_classifier(self, kernel):
        return SVC(kernel = kernel, degree = 2, C = 1) if kernel == "poly" else SVC(kernel = kernel, gamma='scale', C = 1)


    # Evaluate kernel and return mean accuracy, variance and standard variation
    def evaluation_kernel(self, classifier, normalization_flag):
        if normalization_flag: 
            #print("normalized")
            return cross_val_score(classifier, self.norm_X, self.y, cv=CV, n_jobs=-1)
        else: 
            #print("NON normalized")
            return cross_val_score(classifier, self.X, self.y, cv=CV, n_jobs=-1)


	# Score of train test split
    def score(self, classifier, normalization_flag):
        if normalization_flag:
            X_train, X_test, y_train, y_test = self.normalized_data
        else: 
            X_train, X_test, y_train, y_test = self.non_normalized_data
        clf = classifier.fit(X_train, y_train)
        return X_test, y_test, clf.score(X_test, y_test)


    def result(self):
        print("SUPPORT VECTOR MACHINE")

        for norm_flag, p in enumerate(PRINT):
            print(p)
            for k in KERNELS:
                classifier = self.create_classifier(k)
                results = self.evaluation_kernel(classifier, norm_flag)
                X_test, y_test, score = self.score(classifier, norm_flag)
                print_scores(results, k, score)
                print_confusion_matrix(classifier, X_test, y_test, p, k)