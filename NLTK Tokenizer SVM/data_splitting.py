from sklearn.model_selection import train_test_split 

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        random_state=104,  
                                                        test_size=0.2,  
                                                        shuffle=True)
    return X_train, X_test, y_train, y_test