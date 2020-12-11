""" This module is to develop the base
training model with Ridge regression
    This module contain only one function
"""

from warnings import simplefilter
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning


def Ridge_regress(x_train, y_train):
    """ ridge regression.
        X is the independent variables,
        and must be in the format of
        array(can be multidimensional array)
        y is dependent variable,
        and must be in the fromat
        of array (1-D array only)
        this function return
        a model,and also a list
        of coefficient for each independent variable
    """
    simplefilter("ignore", category=ConvergenceWarning)
    model = RidgeClassifier()
    solvers = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    grid = dict(solver=solvers, alpha=alphas)
    con_v = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=grid,
                               n_jobs=-1,
                               cv=con_v,
                               scoring='accuracy', error_score=0)
    grid_result = grid_search.fit(x_train, y_train)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    index_list = []
    if list(means).count(max(means)) == 1:
        index = list(means).index(max(means))
    else:
        for i in range(0, len(means)):
            if means[i] == max(means):
                index_list.append(i)
        index = list(stds).index(min(np.array(stds)[index_list]))
    alpha = params[index]['alpha']
    solver = params[index]['solver']
    model = RidgeClassifier(alpha=alpha, solver=solver).fit(x_train, y_train)
    coeff = list(model.coef_[0]*np.std(x_train, 0))
    return model, coeff
