"""
This module contains functions to prep the data for use in the BioME software.

data_loader reads the OTU and categorical data in and converts the tables
into numpy arrays.

get_one_hot produces one-hot encoded vectors to represent the categorical data
for use in the neural network model, specifically.

split_train_test randomly splits the input data into test and train datasets
(10/90).
"""

import numpy as np
import pandas as pd


def data_loader(path_otu_table, path_metadata, column_number=2):
    """
    Load OTU table and metadata categorical data and convert to numpy array.
    ex. x_data, y_data = data_loader('../bug_OTU_rel.tsv', '../FecesMeta.txt')

    Parameters:
    path_otu_table = string containing path to OTU data table
    path_metadata = string containing path to metadata table with categorical
                    assignments
    column_number = integer specifying column containing categorical data
                    (default=2)

    Returns:
    x_data_np = numpy array containing OTU data (samples x OTU features)
    y_var_np = numpy array containing categorical labels for each sample
    """
    x_data_pd = pd.read_table(path_otu_table, index_col=0, skiprows=1).T
    y_var_pd = pd.read_table(path_metadata, index_col=0,
                             usecols=[0, column_number])
    x_data_np = x_data_pd.to_numpy()
    y_var_np = y_var_pd.to_numpy()
    return x_data_np, y_var_np


def get_one_hot(list_category, y_data, column_number=0):
    """
    Produce one-hot encoded vectors from categorical data
    ex. y_output = get_one_hot(['CD', 'UC', 'IC', 'HC', 'CC'], y_data)

    Parameters:
    list_category = list or 1-D array of strings containing labels for each
                    category
    y_data = numpy array containing categorical assignments for each sample
    column_number = integer specifying which column the categorical data is in
                    (default=0)

    Returns:
    one_hot_output = numpy array containing one-hot encoded classification data
                    (size #samples x #categories)
    """
    number_samples = np.size(y_data, axis=0)
    number_categories = len(list_category)
    one_hot_output = np.zeros((number_samples, number_categories))
    for i in range(number_samples):
        for j in range(number_categories):
            if y_data[i, column_number] == list_category[j]:
                one_hot_output[i, j] += 1
            else:
                pass
    return one_hot_output


def split_train_test(x_data, y_data):
    """
    Function splits data and labels into test and train data (10/90 split)
    ex. x_train, x_test, y_train, y_test = split_train_test(x_data, y_data)

    Parameters:
    x_data = OTU table in form of numpy array with 16S sequences as the columns
    y_data = one-hot encoded categorical labels

    Returns:
    x_train = Random 90% of OTU data
    x_test = Random 10% of OTU data
    y_train = Random 90% of one-hot encoded labels/y array
    y_test = Random 10% of one-hot encoded labels/y array
    """
    number_samples = np.size(x_data, axis=0)
    rand_indices = np.random.permutation(number_samples)
    train_size = round(number_samples*0.9)
    train_indices = rand_indices[0:train_size]
    test_indices = rand_indices[train_size:]
    x_train = x_data[train_indices]
    x_test = x_data[test_indices]
    y_train = y_data[train_indices]
    y_test = y_data[test_indices]
    return x_train, x_test, y_train, y_test
