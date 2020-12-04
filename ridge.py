""" This module is to develop the base training model with Ridge regression 
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

def Ridge_regress(X,y):
    """ ridge regression.
        X is the independent variables, and must be in the format of array(can be multidimensional array)
        y is dependent variable, and must be in the fromat of array (1-D array only)
        this function return a model only
    """
    simplefilter("ignore", category=ConvergenceWarning)
    model = RidgeClassifier()
    solvers = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    grid = dict(solver=solvers,alpha=alphas)
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
    alpha = params[index]['alpha']
    solver = params[index]['solver']
    model=RidgeClassifier(alpha=alpha,solver=solver).fit(X,y)
    return model

    