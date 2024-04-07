from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC

def train_model(X_train, y_train):
    print("--------------------Training model--------------------")
    svm = SVC(kernel="poly", degree=4)
    svm.fit(X_train, y_train)

    return svm

