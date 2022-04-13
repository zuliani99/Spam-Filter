from classifiers.svm_classifier import Svm 
from classifiers.knn_classifier import KNearesNeighbour
from classifiers.nb_classifier import NaiveBayes
from utlis import load_csv, print_scores
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score


def callNb(X, y):
    scores = cross_val_score(NaiveBayes(), X, y, cv = 10)
    print("NAIVE BAYES")
    print_scores(scores)


def callKnn(X, y):
    print("K NEAREST NEIGHBOORS")
    scores = cross_val_score(KNearesNeighbour(), X, y, cv = 10)
    print_scores(scores)


def start():
    # Load Spambase Dataset
    X, y = load_csv()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


    #SVM
    svm = Svm(X,y,X_train,X_test,y_train,y_test)
    svm.result()
    print("-------------------------------------------------------------------") 
    print("\n")

    # NAIVE BAYES
    callNb(X, y)
    print("-------------------------------------------------------------------") 
    print("\n")

    # KNN
    callKnn(X, y)
    
    
if __name__ == "__main__":
   start()