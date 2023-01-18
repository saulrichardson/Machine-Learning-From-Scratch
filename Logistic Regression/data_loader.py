import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_simulated_data():
    """
    Load simulated data and return an X matrix
    for features and Y vector for outcomes
    """

    # load data with pandas
    data = pd.read_csv("simulated_data.csv")
    feature_names = ["intercept"] + list(data.columns)
    feature_names.remove("Y")

    # convert to numpy matrix
    Dmat = data.to_numpy()
    n, d = Dmat.shape

    # separate X matrix and Y vector
    Xmat = Dmat[:, 0:-1]
    Y = Dmat[:, -1]

    # add a column of 1s for intercept term and return
    Xmat = np.column_stack((np.ones(n), Xmat))
    return Xmat, Y, feature_names


def load_breast_cancer_data():

    # load in data with pandas
    data = pd.read_csv("breast_cancer.csv")

    # convert malignant/benign labels to 1/0 binary labels
    Y = np.array([1 if diagnosis=="M" else 0 for diagnosis in data["diagnosis"]])

    # get the feature matrix
    feature_names = [measure for measure in data.columns if "mean" in measure]
    data_features = data[feature_names]
    Xmat = data_features.to_numpy()

    # split into training, validation, testing
    Xmat_train, Xmat_test, Y_train, Y_test = train_test_split(Xmat, Y, test_size=0.2, random_state=1)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train, Y_train, test_size=0.2, random_state=1)

    # standardize the data
    mean = np.mean(Xmat_train, axis=0)
    std = np.std(Xmat_train, axis=0)
    Xmat_train = (Xmat_train - mean)/std
    Xmat_val = (Xmat_val - mean)/std
    Xmat_test = (Xmat_test - mean)/std

    # add a column of ones for the intercept term
    Xmat_train = np.column_stack((np.ones(len(Xmat_train)), Xmat_train))
    Xmat_val = np.column_stack((np.ones(len(Xmat_val)), Xmat_val))
    Xmat_test = np.column_stack((np.ones(len(Xmat_test)), Xmat_test))
    feature_names = ["intercept"] + feature_names

    # return the train/validation/test datasets
    return feature_names, {"Xmat_train": Xmat_train, "Xmat_val": Xmat_val, "Xmat_test": Xmat_test,
                           "Y_train": Y_train, "Y_val": Y_val, "Y_test": Y_test}
