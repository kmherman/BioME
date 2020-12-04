"""
This module contains functions to train two different MLP architectures.

forward_nn1 -- function for forward pass of MLP with single hidden layer
forward_nn3 -- function for forward pass of MLP with three hidden layers
train_nn1 -- function that trains single layer MLP with CV and returns
optimized parameters for model
train_nn3 -- function that trains three layer MLP with CV and returns
optimized parameters for model
"""

import numpy as np
import torch
import torch.nn as nn
from torch import optim


def forward_nn1(x_data, w_0, w_1, b_0, b_1):
    """
    This function performs a forward pass through the single layer MLP
    using inputted parameters on given OTU data.

    Parameters:
    x_data = torch tensor containing OTU data used to make a prediction
    params = torch tensors containing weights and biases for current model

    Returns:
    predictions = torch tensor containing output predictions for x_data
    based on current model
    """
    activ = nn.ReLU()
    layer1 = torch.t(torch.matmul(w_0, torch.t(x_data))) + b_0
    nonlin1 = activ(layer1)
    predictions = torch.t(torch.matmul(w_1, torch.t(nonlin1))) + b_1
    return predictions


def forward_nn3(x_data, w_0, w_1, w_2, w_3, b_0, b_1, b_2, b_3):
    """
    This function performs a forward pass through the three layer MLP
    using inputted parameters on given OTU data.

    Parameters:
    x_data = torch tensor containing OTU data used to make a prediction
    params = torch tensors containing weights and biases for current model

    Returns:
    predictions = torch tensor containing output predictions for x_data
    based on current model.
    """
    activ = nn.ReLU()
    layer1 = torch.t(torch.matmul(w_0, torch.t(x_data))) + b_0
    nonlin1 = activ(layer1)
    layer2 = torch.t(torch.matmul(w_1, torch.t(nonlin1))) + b_1
    nonlin2 = activ(layer2)
    layer3 = torch.t(torch.matmul(w_2, torch.t(nonlin2))) + b_2
    nonlin3 = activ(layer3)
    predictions = torch.t(torch.matmul(w_3, torch.t(nonlin3))) + b_3
    return predictions


def train_nn1(x_train, y_train, h_0=100):
    """
    This function trains the single layer MLP using the Adams optimizer and
    cross entropy loss. K-fold cross validation is also implemented to help
    prevent overfitting of the model.

    Parameters:
    x_train = numpy array of training OTU data
    y_train = numpy array of one-hot encoded classification data
    h_0 = size of single hidden layer (default=100)

    Returns:
    params = torch tensors of optimized weights and biases for MLP.
    """
    features = np.size(x_train, axis=1)
    output_classes = np.size(y_train, axis=1)
    num_train = np.size(x_train, axis=0)

    alpha_0 = 1/np.sqrt(features)
    alpha_1 = 1/np.sqrt(h_0)
    w_0 = np.random.uniform(-alpha_0, alpha_0, (h_0, features))
    w_1 = np.random.uniform(-alpha_1, alpha_1, (output_classes, h_0))
    b_0 = np.random.uniform(-alpha_0, alpha_0, h_0)
    b_1 = np.random.uniform(-alpha_1, alpha_1, output_classes)
    w_0 = torch.tensor(w_0, requires_grad=True)
    w_1 = torch.tensor(w_1, requires_grad=True)
    b_0 = torch.tensor(b_0, requires_grad=True)
    b_1 = torch.tensor(b_1, requires_grad=True)
    x_train_64 = x_train.astype('float64')
    x_train_torch = torch.from_numpy(x_train_64)
    y_train_torch = torch.from_numpy(y_train).type(torch.LongTensor)
    cv_error = 100
    while cv_error > 0.05:
        rand = torch.randperm(num_train)
        x_train_batch = x_train_torch[rand[0:int((8/9)*num_train)]]
        y_train_batch = y_train_torch[rand[0:int((8/9)*num_train)]]
        x_train_cv = x_train_torch[rand[int((8/9)*num_train):]]
        y_train_cv = y_train_torch[rand[int((8/9)*num_train):]]
        train_error = 100
        while train_error > 0.05:
            model_output = forward_nn1(x_train_batch, w_0, w_1, b_0, b_1)
            optimizer = optim.Adam([w_0, w_1, b_0, b_1], lr=0.001,
                                   weight_decay=0.00001)
            loss = nn.CrossEntropyLoss()
            y_target = torch.argmax(y_train_batch, 1)
            loss_iter = loss(model_output, y_target)
            loss_iter.backward()
            optimizer.step()
            optimizer.zero_grad()
            error = 0
            y_train_predict = torch.argmax(model_output, 1)
            y_train_actual = torch.argmax(y_train_batch, 1)
            for i in range(int((8/9)*num_train)):
                if y_train_predict[i] != y_train_actual[i]:
                    error += 1
                else:
                    pass
            train_error = error/int((8/9)*num_train)
        cv_model_output = forward_nn1(x_train_cv, w_0, w_1, b_0, b_1)
        y_cv_predict = torch.argmax(cv_model_output, 1)
        y_cv_actual = torch.argmax(y_train_cv, 1)
        error = 0
        for i in range(len(y_cv_predict)):
            if y_cv_predict[i] != y_cv_actual[i]:
                error += 1
            else:
                pass
        cv_error = error/len(y_cv_predict)
    return (w_0, w_1, b_0, b_1)


def train_nn3(x_train, y_train, h_0=50, h_1=50, h_2=50):
    """
    This function trains the three layer MLP using the Adams optimizer and
    cross entropy loss. K-fold cross validation is also implemented to help
    prevent overfitting of the model.

    Parameters:
    x_train = numpy array of training OTU data
    y_train = numpy array of one-hot encoded classification data
    h_0 = size of first hidden layer (default=50)
    h_1 = size of second hidden layer (default=50)
    h_2 = size of third hidden layer (default=50)

    Returns:
    params = torch tensors of optimized weights and biases for MLP.
    """
    features = np.size(x_train, axis=1)
    output_classes = np.size(y_train, axis=1)
    num_train = np.size(x_train, axis=0)

    alpha_0 = 1/np.sqrt(features)
    alpha_1 = 1/np.sqrt(h_0)
    alpha_2 = 1/np.sqrt(h_1)
    alpha_3 = 1/np.sqrt(h_2)
    w_0 = np.random.uniform(-alpha_0, alpha_0, (h_0, features))
    w_1 = np.random.uniform(-alpha_1, alpha_1, (h_1, h_0))
    w_2 = np.random.uniform(-alpha_2, alpha_2, (h_2, h_1))
    w_3 = np.random.uniform(-alpha_3, alpha_3, (output_classes, h_2))
    w_0 = torch.tensor(w_0, requires_grad=True)
    w_1 = torch.tensor(w_1, requires_grad=True)
    w_2 = torch.tensor(w_2, requires_grad=True)
    w_3 = torch.tensor(w_3, requires_grad=True)
    b_0 = np.random.uniform(-alpha_0, alpha_0, h_0)
    b_1 = np.random.uniform(-alpha_1, alpha_1, h_1)
    b_2 = np.random.uniform(-alpha_2, alpha_2, h_2)
    b_3 = np.random.uniform(-alpha_3, alpha_3, output_classes)
    b_0 = torch.tensor(b_0, requires_grad=True)
    b_1 = torch.tensor(b_1, requires_grad=True)
    b_2 = torch.tensor(b_2, requires_grad=True)
    b_3 = torch.tensor(b_3, requires_grad=True)
    x_train_64 = x_train.astype('float64')
    x_train_torch = torch.from_numpy(x_train_64)
    y_train_torch = torch.from_numpy(y_train).type(torch.LongTensor)
    cv_error = 100
    while cv_error > 0.05:
        rand = torch.randperm(num_train)
        x_train_batch = x_train_torch[rand[0:int((8/9)*num_train)]]
        y_train_batch = y_train_torch[rand[0:int((8/9)*num_train)]]
        x_train_cv = x_train_torch[rand[int((8/9)*num_train):]]
        y_train_cv = y_train_torch[rand[int((8/9)*num_train):]]
        train_error = 100
        while train_error > 0.05:
            model_output = forward_nn3(x_train_batch, w_0, w_1, w_2, w_3, b_0,
                                       b_1, b_2, b_3)
            optimizer = optim.Adam([w_0, w_1, w_2, w_3, b_0, b_1, b_2, b_3],
                                   lr=0.001, weight_decay=0.00001)
            loss = nn.CrossEntropyLoss()
            y_target = torch.argmax(y_train_batch, 1)
            loss_iter = loss(model_output, y_target)
            loss_iter.backward()
            optimizer.step()
            optimizer.zero_grad()
            error = 0
            y_train_predict = torch.argmax(model_output, 1)
            y_train_actual = torch.argmax(y_train_batch, 1)
            for i in range(int((8/9)*num_train)):
                if y_train_predict[i] != y_train_actual[i]:
                    error += 1
                else:
                    pass
            train_error = error/int((8/9)*num_train)
        cv_model_output = forward_nn3(x_train_cv, w_0, w_1, w_2, w_3, b_0,
                                      b_1, b_2, b_3)
        y_cv_predict = torch.argmax(cv_model_output, 1)
        y_cv_actual = torch.argmax(y_train_cv, 1)
        error = 0
        for i in range(len(y_cv_predict)):
            if y_cv_predict[i] != y_cv_actual[i]:
                error += 1
            else:
                pass
        cv_error = error/len(y_cv_predict)
    return (w_0, w_1, w_2, w_3, b_0, b_1, b_2, b_3)
