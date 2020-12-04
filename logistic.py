""" This module is to develop the base training model with losgistic regression 
    This module contain only one function 
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
from warnings import simplefilter    
from sklearn.exceptions import ConvergenceWarning


def logistic_regress(X,y):
    """ logistic regression.
        X is the independent variables, and must be in the format of array(can be multidimensional array)
        y is dependent variable, and must be in the fromat of array (1-D array only)
        this function return a model only
    """
    simplefilter("ignore", category=ConvergenceWarning)
    model = LogisticRegression()
    solvers=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    penalties=['l1','l2','elasticnet','none']
    c_values =[100, 10, 1.0, 0.1, 0.01]
    grid = dict(solver=solvers,penalty=penalties,C=c_values)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    grid_result = grid_search.fit(X,y)
    means = grid_result.cv_results_['mean_test_score'] # 1-D array
    stds = grid_result.cv_results_['std_test_score'] # 1-D array
    params = grid_result.cv_results_['params'] # this return a List of dictionary 
    index_list=[]
    if list(means).count(max(means))==1:
        index=list(means).index(max(means))
    else:
        for i in range(0,len(means)):
            if means[i]==max(means):index_list.append(i)
        index = list(stds).index(min(np.array(stds)[index_list]))
    regulation = params[index]['C']
    penalty = params[index]['penalty']
    solver = params[index]['solver']
    model = LogisticRegression(random_state=0,penalty=penalty,solver=solver,C=regulation).fit(X, y)
    return model