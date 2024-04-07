import sklearn.metrics as metrics

def display_results(y_true, y_pred):
    print('\nAccuracy:', metrics.accuracy_score(y_true=y_true, y_pred=y_pred))
    print('Precision:', metrics.precision_score(y_true=y_true, y_pred=y_pred))
    print('Recall:', metrics.recall_score(y_true=y_true, y_pred=y_pred))
    print('F-measure:', metrics.f1_score(y_true=y_true, y_pred=y_pred))

