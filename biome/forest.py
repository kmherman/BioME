""" This module is to develop the base training model with random forest
    This module contain only one function
    warning: random forest algorthm may take more 10 minute to run
"""

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import numpy as np


def random_forest(x_train, y_train):
    """ random forest
        X is the independent variables,
        and must be in the format of
        array(can be multidimensional array)
        y is dependent variable, and must be
        in the fromat of array (1-D array only)
        this function return a model,
        and a list of coefficient which
        can be consider as the feature importance
    """
    simplefilter("ignore", category=ConvergenceWarning)
    model = RandomForestClassifier()
    n_estimators = [10, 100]
    max_features = [ 'sqrt']
    criterion = ['gini']
    grid = dict(n_estimators=n_estimators,
                criterion=criterion,
                max_features=max_features)
    con_v = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model,
                               param_grid=grid,
                               n_jobs=-1, cv=con_v,
                               scoring='accuracy',
                               error_score=0)
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
    n_estimator = params[index]['n_estimators']
    max_feature = params[index]['max_features']
    criterion = params[index]['criterion']
    model = RandomForestClassifier(random_state=0,
                                   n_estimators=n_estimator,
                                   max_features=max_feature,
                                   criterion=criterion).fit(x_train, y_train)
    coeff = list(model.feature_importances_)
    return model, coeff
