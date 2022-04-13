import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

RESULT = "files/result.txt"
CV = 10
KERNELS = ["linear", "poly", "rbf"]
svm_list = []
svm_angular_kernel_list = []

class Svm:

    def __init__ (self, X, Y):
        self.X = X
        self.Y = Y

    # calculate the term frequency of each word in each mail
    def tf_idf(self):
        idf = np.log10(self.X.shape[0] / (self.X != 0).sum(0))
        return self.X / 100.0 * idf

    # evaluate kernel and return mean accuracy, variance and standard variation
    def evaluation_kernel(self, kernel):
        if kernel == "poly": classifier = SVC(kernel=kernel, degree=2, C=1.0)
        else: classifier = SVC(kernel=kernel, C=1.0)
        svm_list.append(classifier)
        ten_way_CV_score = cross_val_score(classifier, self.X, self.Y, cv=CV, n_jobs=-1)
        mean = ten_way_CV_score.mean()
        var = ten_way_CV_score.var()
        std = ten_way_CV_score.std()

        return [str(mean), str(var), str(std)]

    def classifier_score(self, x_train, y_train):
        fit = self.fit(x_train, y_train)
        return fit.score(x_train, y_train)

    def result(self):
        file = open(RESULT, "w")
        file.write("SVM CLASSIFIER \n")
        file.write("\n")

        # kernels without angular kernel evaluation
        file.write("Measures of kernels without angular kernel evaluation: \n")
        for k in KERNELS:
            results = self.evaluation_kernel(k)
            write_result(file, k, results)

        file.write("\n")

        # kernels with angular kernel evaluation
        file.write("Measures of kernels with angular kernel evaluation: \n")
        for k in KERNELS:
            results = self.evaluation_kernel(k)
            write_result(file, k, results)

            

def write_result(file, kernel, results):
    file.write("\n")
    file.write(f"---- {kernel} kernel " + " ---- \n")
    file.write(f"Mean Accuracy: {str(results[0])}" + "\n")
    file.write(f"Variance Accuracy: {str(results[1])}" + "\n")
    file.write(f"Standard Variation Accuracy: {str(results[2])}" + "\n")
