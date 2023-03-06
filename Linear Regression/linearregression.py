#!/usr/bin/env python

import pandas as pd
import numpy as np
from data_loader import load_simulated_data, load_insurance_data
import statsmodels.api as sm


def mse(Y, Yhat):
    """
    Calculates mean squared error between ground-truth Y and predicted Y

    """

    return np.mean((Y-Yhat)**2)

def rsquare(Y, Yhat):
    """
    Implementation of the R squared metric based on ground-truth Y and predicted Y
    """

    avg = np.mean(Y)
    t = np.sum((Y - Yhat)**2)
    b = np.sum((Y - avg)**2)
    return 1 - (t/b)

def tol(prog, tol):
    #not enough iterations to compare change in theta so return False
    if len(prog) <= 2:
        return False
    else:
        sumDiff = []

        # GO through theta and calculate absolute difference and sum together
        for a in range(len(prog[-1])):
            diff = prog[-1][a] - prog[-2][a]
            sumDiff.append(np.absolute(diff))
        sumDiff = np.array(sumDiff, dtype = float)

        # return whether mean abs difference is less than tolerance value
        return np.mean(sumDiff) < tol



class LinearRegression:
    """
    Class for linear regression
    """

    def __init__(self, learning_rate=0.01):
        """
        Constructor for the class. Learning rate is
        any positive number controlling step size of gradient descent.
        """

        self.learning_rate = learning_rate
        self.theta = None # theta is initialized once we fit the model

    def _calculate_gradient(self, Xmat, Y, theta_p, h=1e-5):
        """
        Helper function for computing the gradient at a point theta_p.
        """
        #analytical method
        #(2/n) * E(Yhat_i-Y_i) * derivativeOf(Yhat_i-Y_i)

        #ex: for d/dtheta_1
        #(2/n) * E(Yhat_i-Y_i) * x_1

        n, d = Xmat.shape
        pred = Xmat @ np.transpose(theta_p) #Yhat
        diff = pred - Y  #Y-Yhat

        #derivative of (Y-Yhat) is just going to be whatever x corresponds
        #to the theta we're differentiating by. We're taking that x_i and multipying it
        #by the corresponding diff and summing across i. This is just matrix
        #muliplciation. So we just multipy the diff matrix and the matrix of X's
        #to get E(Y_i-Yhat_i) * derivativeOf(Y-Yhat)_i and then all we have to do is
        #multipy by (2/n).

        working = diff @ (Xmat)
        working = working * (2/n)

        return working




    def fit(self, Xmat, Y, max_iterations=1000000, tolerance=1e-10, verbose=False):
        """
        Fit a linear regression model using training data Xmat and Y.
        """

        # get dimensions of the matrix
        n, d = Xmat.shape #n -> obs, d -> features

        # initialize a guess for theta
        theta = np.random.uniform(-5.0, 5.0, d)

        progress = [theta.copy()]
        mseP = []
        while (max_iterations > 0 and not tol(progress, tolerance)):
            max_iterations -= 1
            mseP.append(mse(Y, Xmat @ np.transpose(theta)))
            theta -= (self.learning_rate * self._calculate_gradient(Xmat, Y, theta))
            progress.append(theta.copy())


        # TODO: Implement code that performs gradient descent until "convergence"
        # i.e., until max_iterations or until the change in theta measured by mean absolute difference
        # is less than the tolerance argument


        # set the theta attribute of the model to the final value from gradient descent
        # self.theta = theta_new.copy() why are there two thetas?
        self.theta = theta.copy()


def main():
    """
    Do not edit this function. This function is used for grading purposes only.
    """

    np.random.seed(0)

    #################
    # Simulated data
    #################
    Xmat, Y, feature_names = load_simulated_data()
    # Xmat = sm.add_constant(Xmat, prepend=False)


    model = LinearRegression()
    model.fit(Xmat, Y)
    Yhat = Xmat @ model.theta
    print("Simulated data results:\n" + "-"*4)
    print("Simulated data fitted weights", {feature_names[i]: round(model.theta[i], 2) for i in range(len(feature_names))})
    print("R squared simulated data", rsquare(Y, Yhat), "\n")

    #################
    # Insurance data
    #################
    Xmat_train, Y_train, Xmat_test, Y_test, feature_names = load_insurance_data()
    model = LinearRegression()

    model.fit(Xmat_train, Y_train) # only use training data for fitting
    Yhat_test = Xmat_test @ model.theta # evaluate on the test data
    print("Insurance data results:\n" + "-"*4)
    print("Insurance data fitted weights", {feature_names[i]: round(model.theta[i], 2) for i in range(len(feature_names))})
    print("R squared insurance data", rsquare(Y_test, Yhat_test))


if __name__ == "__main__":
    main()
