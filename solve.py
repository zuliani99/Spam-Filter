from classifiers.svm_classifier import Svm 
from classifiers.knn_classifier import callKnn
from classifiers.nb_classifier import callNb
from utlis import load_csv
from sklearn.model_selection import train_test_split

def start():
    # Load Spambase Dataset
    X, y = load_csv()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    #SVM
    Svm(X, y, X_train, X_test, y_train, y_test).result()
    
    # NAIVE BAYES
    callNb(X, y, X_train, X_test, y_train, y_test)

    # KNN
    callKnn(X, y, X_train, X_test, y_train, y_test)
    
    
if __name__ == "__main__":
   start()