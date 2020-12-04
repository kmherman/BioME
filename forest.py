""" This module is to develop the base training model with losgistic regression 
    This module contain only one function 
    warning: random forest algorthm may take more 10 minute to run
"""

import numpy as np
from numpy.linalg import svd
import math
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import random
import itertools 
from sklearn import datasets
from sklearn import svm
from sklearn.linear_model import LogisticRegression 
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler  
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from warnings import simplefilter    
from sklearn.exceptions import ConvergenceWarning

def Ridge_regress(X,y):
    """ ridge regression.
        X is the independent variables, and must be in the format of array(can be multidimensional array)
        y is dependent variable, and must be in the fromat of array (1-D array only)
        this function return a model, and a list of coefficient which can be consider as the feature importance
    """
    simplefilter("ignore", category=ConvergenceWarning)
    model = RandomForestClassifier()
    n_estimators = [10, 100, 1000]
    max_features = ['auto','sqrt', 'log2']
    criterion=['gini', 'entropy']
    grid = dict(n_estimators=n_estimators,criterion=criterion,max_features=max_features)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X, y)
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    index_list=[]
    if list(means).count(max(means))==1:
        index=list(means).index(max(means))
    else:
        for i in range(0,len(means)):
            if means[i]==max(means):index_list.append(i)
        index = list(stds).index(min(np.array(stds)[index_list]))
    n_estimator = params[index]['n_estimators']
    max_feature = params[index]['max_features']
    criterion = params[index]['criterion']
    model = RandomForestClassifier(random_state=0,n_estimators=n_estimator,max_features=max_feature,criterion=criterion).fit(X,y)
    coeff = list(model.feature_importances_)
    return model, coeff

    