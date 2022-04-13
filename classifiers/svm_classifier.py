import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

CV = 10
KERNELS = ["linear", "poly", "rbf"]
svm_list = []
svm_angular_kernel_list = []


class Svm:

    def __init__ (self, X, Y, X_train, X_test, y_train, y_test):
        self.X = self.tf_idf(X)
        self.Y = Y
        self.X_train = self.tf_idf(X_train)
        self.X_test = self.tf_idf(X_test)
        self.y_train = y_train
        self.y_test = y_test


    # calculate the term frequency of each word in each mail
    def tf_idf(self, X):
        idf = np.log10(X.shape[0] / (X != 0).sum(0))
        return X / 100.0 * idf


    # normalize dataset
    def normalization(self,X):
        norms = np.sqrt(((X+1e-128) ** 2).sum(axis=1, keepdims=True))
        return np.where(norms > 0.0, X / norms, 0.)


    def create_classifier(self, kernel):
        if kernel == "poly": classifier = SVC(kernel=kernel, degree=2, C=1.0)
        else: classifier = SVC(kernel=kernel, C=1.0)
        return classifier


    # evaluate kernel and return mean accuracy, variance and standard variation
    def evaluation_kernel(self, classifier, flag):
        svm_list.append(classifier)
        if flag: ten_way_CV_score = cross_val_score(classifier, self.normalization(self.X), self.Y, cv=CV, n_jobs=-1)
        else: ten_way_CV_score = cross_val_score(classifier, self.X, self.Y, cv=CV, n_jobs=-1)
        
        min = ten_way_CV_score.min()
        max = ten_way_CV_score.max()
        mean = ten_way_CV_score.mean()
        var = ten_way_CV_score.var()
        std = ten_way_CV_score.std()

        return [str(min), str(max), str(mean), str(var), str(std)]


    def score(self, classifier, flag):
        if flag: X_train=self.normalization(self.X_train)
        else: X_train=self.X_train
        
        fit = classifier.fit(X_train, self.y_train)
        return fit.score(X_train, self.y_train)


    def result(self):
        print("SVM CLASSIFIER \n")
        print("\n")

        # kernels without angular kernel evaluation
        print("Measures of kernels without angular kernel evaluation: \n")
        for k in KERNELS:
            classifier=self.create_classifier(k)
            results = self.evaluation_kernel(classifier,0)      # flag=0 -> without data normalization
            score = self.score(classifier,0)                    # flag=0 -> without data normalization
            write_result(k, results, score)
            
        print("\n")

        # kernels with angular kernel evaluation
        print("Measures of kernels with angular kernel evaluation: \n")
        for k in KERNELS:
            classifier=self.create_classifier(k)
            results = self.evaluation_kernel(classifier,1)      # flag=1 -> with data normalization
            score = self.score(classifier,1)                    # flag=1 -> with data normalization
            write_result(k, results, score)




def write_result(kernel, results, score):
    print("\n")
    print(f"---- {kernel} kernel " + " ---- \n")
    print(f"Cross Val - Min Accuracy: \t\t{str(round(results[0]))}" + "\n")
    print(f"Cross Val - Max Accuracy: \t\t{str(round(results[1]))}" + "\n")
    print(f"Cross Val - Mean Accuracy: \t\t{str(round(results[2]))}" + "\n")
    print(f"Cross Val - Variance: \t\t\t{str(round(results[3]))}" + "\n")
    print(f"Cross Val - Standard Deviation: \t{str(round(results[4]))}" + "\n")
    print("\n")
    print(f"Cross Val - Score Accuracy: \t\t{str(round(score))}" + "\n")