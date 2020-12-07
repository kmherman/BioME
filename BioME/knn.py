import sys
import pandas as pd
from prep_split_data import get_one_hot
from prep_split_data import split_train_test
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

def knn(x_train, y_train):
    """
    Function performs KNN on dataset
    Parameters:
    x_train = independent variables
    y_train = dependent variable
    Returns:
    knn.fit(X_test, y_test) - fitted KNN model
    """
    model = KNeighborsClassifier()
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(x_train)
    X_test = pca.transform(x_test)
    '# Create KNN classifier'
    N = int(input("enter number of neighbors: "))
    knn = KNeighborsClassifier(n_neighbors=N)
    '# Fit the classifier to the data'
    knn.fit(X_train, y_train)

    '# shows model predictions on the test data'
    #knn.predict(X_test)

    '# check accuracy of our model on the test data'
    #knn.score(X_test, y_test)
    
    return model

