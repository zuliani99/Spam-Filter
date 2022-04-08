import pandas as pd

def readData():
    data = pd.read_csv("spambase.data", "," ,index_col = False)
    return data.iloc[:, :54], data.iloc[:,57]

if __name__ == "__main__":
    X, Y = readData()
    #SVM
    #NAIVE BAYES
    #KNN