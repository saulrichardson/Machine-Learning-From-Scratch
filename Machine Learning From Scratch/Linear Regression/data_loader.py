import pandas as pd
import numpy as np


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

def load_insurance_data():
    """
    Load insurance data, standardize the continuous variables,
    produce dummy variables for categorical ones, and split
    the data into training and testing sets
    """

    # load data with pandas and create dummy variables for categorical inputs
    data = pd.read_csv("insurance.csv")
    data = pd.get_dummies(data)
    feature_names = ["intercept"] + list(data.columns)
    feature_names.remove("charges")

    # pre-processing
    Dmat = data.to_numpy()

    # standardize first 4 variables corresponding to continuous variables
    Dcont = Dmat[:, 0:4]
    Dcont = (Dcont - Dcont.mean(axis=0))/Dcont.std(axis=0)

    # add a column of 1s for the intercept term
    Xmat = np.column_stack((np.ones(len(Dmat)), Dcont[:, 0:3], Dmat[:, 4:]))

    # extract outcome vector
    Y = Dcont[:, 3]
    
    # produce a train-test split
    n = len(Xmat)
    Xmat_train = Xmat[0:int(0.8*n), :]
    Xmat_test = Xmat[int(0.8*n):, :]
    Y_train = Y[0:int(0.8*n)]
    Y_test = Y[int(0.8*n):]

    return Xmat_train, Y_train, Xmat_test, Y_test, feature_names
