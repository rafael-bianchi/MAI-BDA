import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

def param_selection(clf, param_grid, X, y, nfolds, print_averages=True):
    grid_search = GridSearchCV(clf, param_grid, return_train_score = print_averages, cv=nfolds, n_jobs =-1)
    grid_search.fit(X, y)
    
    if(print_averages):
        means = grid_search.cv_results_['mean_test_score']
        stds = grid_search.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))    

    # Print best score and parameters
    print("%0.3f for %r" % (grid_search.best_score_, grid_search.best_params_))
    return grid_search.best_estimator_, grid_search.best_params_