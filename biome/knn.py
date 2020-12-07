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
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(x_train)
    '# Create KNN classifier'
    N = int(input("enter number of neighbors: "))
    knn = KNeighborsClassifier(n_neighbors=N)
    '# Fit the classifier to the data'
    model = knn.fit(X_train, y_train)
    return model
