from sklearn.metrics import accuracy_score

def evaluate_model(model, X_test, y_test):
    print("--------------------Evaluating model--------------------")
    y_pred = model.predict(X_test)
    return y_pred