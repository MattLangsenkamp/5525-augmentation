import xgboost as xgb

def train_classifier(X, y):
    
    train = xgb.DMatrix(X, label = y)
    
    param = {'max_depth':8, 'eta':1, 'silent':1, 'objective': 'multi:softprob','num_class':20}

    # specify validations set to watch performance
    #watchlist = [(dtest_label, 'eval'), (dtrain, 'train')]
    num_round = 2
    bst = xgb.train(param, train, num_round) #, watchlist

    return bst
