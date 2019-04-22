import xgboost as xgb
from sklearn.metrics import confusion_matrix
import numpy as np

def test_classifier(classifier, test_X, test_y):
    
    X = xgb.DMatrix(test_X)
    
    preds = np.argmax(classifier.predict(X), axis=1)
    
    print(preds)
    con_mat = confusion_matrix(test_y, preds, labels=None, sample_weight=None)
    acc = sum(1*(test_y == preds))/len(preds)
    return acc, con_mat