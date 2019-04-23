import xgboost as xgb
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import numpy as np

def test_classifier_xgb(classifier, test_X, test_y):
    
    X = xgb.DMatrix(test_X)
    
    preds = np.argmax(classifier.predict(X), axis=1)
    con_mat = confusion_matrix(test_y, preds, labels=None, sample_weight=None)
    acc = sum(1*(test_y == preds))/len(preds)
    return acc, con_mat

def test_classifier_bayes(classifier, test_X, test_y):
    
    preds = classifier.predict(test_X)
    con_mat = confusion_matrix(test_y, preds, labels=None, sample_weight=None)
    acc = sum(1*(test_y == preds))/len(preds)
    return acc, con_mat