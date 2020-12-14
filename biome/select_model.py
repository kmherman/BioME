"""
This module performs calls the requested ML algorithms and trains them, scores
them, ranks them, and returns the model that performs the best on the data. It
also contains a function to perform a prediction using the "best" model.

get_trained_models -- trains each model listed (or all) by calling in each
ML algorithm and returning each trained model.
evaluate_rank_models -- evaluates each selected model on test set and ranks
each model by its score. Prints out table of ranked models and score.
get_prediction -- utilizes best-performing model to make a predicted disease
classification given a query data point.
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

from .train_mlp import forward_nn1
from .train_mlp import forward_nn3
from .train_mlp import train_nn1
from .train_mlp import train_nn3
from .logistic import logistic_regress
from .ridge import Ridge_regress
from .SVC import get_SVC
from .dtree import decision_tree
from .knn import knn
from .forest import random_forest
from .naive_bayes import GNB


def get_trained_models(x_train, y_train, list_models):
    """
    This function trains each of the models specified using the training
    data and returns each model.

    Parameters:
    x_train = numpy array containing OTU training data
    y_train = numpy array containing one-hot encoded categorical data
    list_models = a list of strings specifying which models should be trained

    Returns:
    trained_models = list of all trained models
    
    Raises exception when a model selected is not an available ML algorithm
    in BioME.
    """
    available_models = ['mlp1', 'mlp3', 'lr', 'rr', 'svc', 'dtree', 'knn',
                        'rf', 'gnb']
    trained_models = []
    y_train_num = np.argmax(y_train, 1)
    if 'all' in list_models:
        list_models_all = available_models
    else:
        list_models_all = list_models
    if (set(list_models_all).issubset(set(available_models))):
        pass
    else:
        raise ValueError('A model selected is not an available ML algorithm.')
    if 'mlp1' in list_models_all:
        params = train_nn1(x_train, y_train)
        trained_models.append(params)
    else:
        pass
    if 'mlp3' in list_models_all:
        params = train_nn3(x_train, y_train)
        trained_models.append(params)
    else:
        pass
    if 'lr' in list_models_all:
        lr_model, coeff = logistic_regress(x_train, y_train_num)
        trained_models.append(lr_model)
    else:
        pass
    if 'rr' in list_models_all:
        rr_model, coeff = Ridge_regress(x_train, y_train_num)
        trained_models.append(rr_model)
    else:
        pass
    if 'svc' in list_models_all:
        svc_model = get_SVC(x_train, y_train_num)
        trained_models.append(svc_model)
    else:
        pass
    if 'dtree' in list_models_all:
        dtree_model = decision_tree(x_train, y_train_num)
        trained_models.append(dtree_model)
    else:
        pass
    if 'knn' in list_models_all:
        knn_model = knn(x_train, y_train_num)
        trained_models.append(knn_model)
    else:
        pass
    if 'rf' in list_models_all:
        rf_model, coeff = random_forest(x_train, y_train_num)
        trained_models.append(rf_model)
    else:
        pass
    if 'gnb' in list_models_all:
        gnb_model = GNB(x_train, y_train_num)
        gnb_model = GNB.get_GNB(x_train, y_train_num)
        trained_models.append(gnb_model)
    else:
        pass
    return trained_models


def evaluate_rank_models(x_train, y_train, x_test, y_test, list_models):
    """
    This model evaluates the trained models on the test data and ranks them
    by accuracy score.

    Parameters:
    x_train = numpy array of training OTU data
    y_train = numpy array of categorical data
    x_test = numpy array of testing OTU data
    y_test = numpy array of corresponding test categorical data
    list_models = list containing abbreviations for models to test

    Returns:
    (model_name, best_model) = tuple containing title of best-performing model
    and the returned, fitted model.

    prints: List of algorithms and ROC-AUC score ranks in descending order.
    files: produces two files that save the model name and the
    best-performing model itself.
    """
    trained_models = get_trained_models(x_train, y_train, list_models)

    x_test_64 = x_test.astype('float64')
    x_test_torch = torch.from_numpy(x_test_64)
    score_list = []
    y_test_num = np.argmax(y_test, 1)
    count = 0
    if 'all' in list_models:
        list_models_all = ['mlp1', 'mlp3', 'lr', 'rr', 'svc', 'dtree', 'knn',
                           'rf', 'gnb']
    else:
        list_models_all = list_models
    if 'mlp1' in list_models_all:
        params = trained_models[count]
        model_out = forward_nn1(x_test_torch, params[0], params[1], params[2],
                                params[3])
        model_output_np = model_out.detach().numpy()
        model_output = np.argmax(model_output_np, 1)
        score = accuracy_score(y_test_num, model_output)
        count += 1
        score_list.append(['MLP (single hidden layer)',
                           score.astype('float16')])
    else:
        pass
    if 'mlp3' in list_models_all:
        params3 = trained_models[count]
        model_out = forward_nn3(x_test_torch, params3[0], params3[1],
                                params3[2], params3[3], params3[4],
                                params3[5], params3[6], params3[7])
        model_output_np = model_out.detach().numpy()
        model_output = np.argmax(model_output_np, 1)
        score = accuracy_score(y_test_num, model_output)
        count += 1
        score_list.append(['MLP (three hidden layers)',
                           score.astype('float16')])
    else:
        pass
    if 'lr' in list_models_all:
        lr_model = trained_models[count]
        model_out = lr_model.predict(x_test).reshape(-1, 1)
        score = accuracy_score(y_test_num, model_out)
        count += 1
        score_list.append(['Logistic Regression',
                           score.astype('float16')])
    else:
        pass
    if 'rr' in list_models_all:
        rr_model = trained_models[count]
        model_out = rr_model.predict(x_test).reshape(-1, 1)
        score = accuracy_score(y_test_num, model_out)
        count += 1
        score_list.append(['Ridge Classifier',
                           score.astype('float16')])
    else:
        pass
    if 'svc' in list_models_all:
        svc_model = trained_models[count]
        model_out = svc_model.predict(x_test).reshape(-1, 1)
        score = accuracy_score(y_test_num, model_out)
        count += 1
        score_list.append(['Support Vector Classifier',
                           score.astype('float16')])
    else:
        pass
    if 'dtree' in list_models_all:
        dtree_model = trained_models[count]
        model_out = dtree_model.predict(x_test).reshape(-1, 1)
        score = accuracy_score(y_test_num, model_out)
        count += 1
        score_list.append(['Decision Tree',
                           score.astype('float16')])
    else:
        pass
    if 'knn' in list_models_all:
        knn_model = trained_models[count]
        pca = PCA(n_components=2)
        pca.fit_transform(x_train)
        X_test = pca.transform(x_test)
        model_out = knn_model.predict(X_test).reshape(-1, 1)
        score = accuracy_score(y_test_num, model_out)
        count += 1
        score_list.append(['k-nearest neighbors',
                           score.astype('float16')])
    else:
        pass
    if 'rf' in list_models_all:
        rf_model = trained_models[count]
        model_out = rf_model.predict(x_test).reshape(-1, 1)
        score = accuracy_score(y_test_num, model_out)
        count += 1
        score_list.append(['Random Forest',
                           score.astype('float16')])
    else:
        pass
    if 'gnb' in list_models_all:
        gnb_model = trained_models[count]
        model_out = gnb_model.predict(x_test).reshape(-1, 1)
        score = accuracy_score(y_test_num, model_out)
        count += 1
        score_list.append(['Gaussian Naive Bayes',
                          score.astype('float16')])
    else:
        pass
    sorted_list = sorted(score_list, key=lambda x: x[1], reverse=True)
    max_val = max(score_list, key=lambda x: x[1])
    index_max = score_list.index(max_val)

    reformat_list = []
    reformat_list.append('\033[4mModel: Accuracy score\033[0m')
    for element in sorted_list:
        reformat_list.append(element[0] + ': ' + str(element[1]))
    best_model_score = sorted_list[0]
    print(*reformat_list, sep='\n')
    print('')
    print('The best performing model is: ' + str(best_model_score[0]))
    print('')
    model_name = best_model_score[0]
    best_model = trained_models[index_max]
    return (model_name, best_model)


def get_prediction(query_data, model_name, best_model, category_labels):
    """
    This function fits the model to a query point or set of query points
    to predict the disease classification with the model.

    Parameters:
    query_path = string that defines the relative path to the .tsv query
    data point.
    category_list = string of categorical labels that will be converted
    to a list.

    Returns:
    prediction = string defining the predicted categorical data.

    prints: predicted disease classification (from provided categorical list).
    """
    query_data_64 = query_data.astype('float64')
    query_data_torch = torch.from_numpy(query_data_64)
    if model_name == 'MLP (single hidden layer)':
        model_out = forward_nn1(query_data_torch, best_model[0],
                                best_model[1], best_model[2], best_model[3])
        model_out = model_out.detach().numpy()
        predict_number = np.argmax(model_out)
    elif model_name == 'MLP (three hidden layers)':
        model_out = forward_nn3(query_data_torch, best_model[0],
                                best_model[1], best_model[2], best_model[3],
                                best_model[4], best_model[5], best_model[6],
                                best_model[7])
        model_out = model_out.detach().numpy()
        predict_number = np.argmax(model_out)
    else:
        model_out = best_model.predict(query_data)
        predict_number = model_out[0]
    prediction = category_labels[predict_number]
    print('The predicted category is: ' + str(prediction))
    return prediction
