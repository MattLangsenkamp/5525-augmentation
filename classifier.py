import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB

def train_classifier_xgb(X, y):
    
    train = xgb.DMatrix(X, label = y)
    
    param = {'max_depth':35, 'eta':1, 'silent':1, 'objective': 'multi:softprob','num_class':20}

    # specify validations set to watch performance
    #watchlist = [(dtest_label, 'eval'), (dtrain, 'train')]
    num_round = 2
    bst = xgb.train(param, train, num_round) #, watchlist

    return bst

def train_classifier_bayes(X, y):
    
    clf = MultinomialNB()
    clf.fit(X, y)
    
    return clf
