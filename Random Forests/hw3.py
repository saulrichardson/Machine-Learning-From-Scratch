#!/usr/bin/env python
import math
import numpy as np
import pandas as pd
from data_loader import load_simulated_data, load_transplant_data
from sklearn.model_selection import train_test_split


def accuracy(Y, Yhat):
    """
    Function for computing accuracy

    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """

    return np.sum(Y==Yhat)/len(Y)

def entropy(data, outcome_name):
    """
    Compute entropy of assuming the outcome is binary

    data: a pandas dataframe
    outcome_name: a string corresponding to name of the outcome varibale
    """

    # compute p(Y=1)
    proba_Y1 = np.mean(data[outcome_name])

    # check if entropy is 0
    if proba_Y1 == 0 or proba_Y1 == 1:
        return 0

    # compute and return entropy
    return -(proba_Y1)*np.log2(proba_Y1) - (1-proba_Y1)*np.log2(1-proba_Y1)

def weighted_entropy(data1, data2, outcome_name):
    """
    Calculate the weighted entropy of two datasets

    data1: a pandas dataframe
    data2: a pandas dataframe
    outcome_name: a string corresponding to name of the outcome varibale
    """

    return (( len(data1) * entropy(data1, outcome_name)) + ( len(data2) * entropy(data2, outcome_name) ))/ (len(data1) + len(data2))

class Vertex:
    """
    Class for defining a vertex in a decision tree
    """

    def __init__(self, feature_name=None, threshold=None, prediction=None):

        self.left_child = None
        self.right_child = None
        self.feature_name = feature_name # name of feature to split on
        self.threshold = threshold # threshold of feature to split on
        self.prediction = prediction # predicted value -- applies only to leaf nodes


class DecisionTree:
    """
    Class for building decision trees
    """

    def __init__(self, max_depth=np.inf):

        self.max_depth = max_depth
        self.root = None


    # helper method to find best value to split for given column in data
    def _best_threshold(self, data, column, outcome_name):
        tracker = math.inf
        threshold_value = None
        possible = data[column].unique()

        # assumes set of possible values is greater than one or else
        # split1 will be empty
        for p in possible:
            split1 = data[data[column] < p]
            split2 = data[data[column] >= p]
            entropy = weighted_entropy(split1, split2, outcome_name)
            if entropy < tracker:
                tracker = entropy
                threshold_value = p

            if entropy == 0:
                break #cannot do any better
        # print(data)
        # print(possible)
        return threshold_value, entropy


    def _get_best_split(self, data, outcome_name):
        """
        Method to compute the best split of the data to minimize entropy

        data: pandas dataframe
        outcome_name: a string corresponding to name of the outcome varibale

        Returns
        ------
        A tuple consisting of:
        (i) String corresponding to name of the best feature
        (ii) Float corresponding to value to split the feature on
        (iii) pandas dataframe consisting of subset of rows of data where best_feature < best_threshold
        (iv) pandas dataframe consisting of subset of rows of data where best_feature >= best_threshold
        """

        best_entropy = entropy(data, outcome_name)
        best_feature = None
        best_threshold = 0
        data_left = None
        data_right = None


        for col in data.columns:
            if col == outcome_name:
                continue

            best_t, best_e = self._best_threshold(data, col, outcome_name)

            # use <= just in case initial data has same value
            # for outcome, i.e., zero entropy. If that occurs,
            # we still need to initialize the variables.
            # since entropy is already zero in this case, any arbitrary
            # split of data will yield zero entropy so

            if best_e <= best_entropy:
                best_entropy = best_e
                best_threshold = best_t
                best_feature = col
                data_left = data[data[col] < best_t]
                data_right = data[data[col] >= best_t]

            if best_entropy == 0:
                break #cannot do any better

        return best_feature, best_threshold, data_left, data_right




    def _build_tree(self, data, outcome_name, curr_depth=0):
        """
        Recursive function to build a decision tree. Refer to the HW pdf
        for more details on the implementation of this function.

        data: pandas dataframe
        outcome_name: a string corresponding to name of the outcome varibale
        curr_depth: integer corresponding to current depth of the tree
        """
        # assuming outcome is bianry
        # no need to split if outcome is all same

        # print("a",set(data[outcome_name]))
        check = data[outcome_name].to_numpy()
        # print(data)
        if (check[0] == check).all():
            return Vertex(prediction = data[outcome_name].iloc[0])

        if curr_depth >= self.max_depth:
            # low deth decision tree may have leaf where not all outcome values are same
            # if we can't split anymore, we aren't gaurenteed to have data
            # that is pure, so let's just return the most common value
            return Vertex(prediction = data[outcome_name].mode()[0])
        else:
            # print("d", data)
            best_f, best_t, data_l, data_r = self._get_best_split(data, outcome_name)
            # print("l",set(data_l[outcome_name]))
            # print("r",set(data_r[outcome_name]))
            # print(best_f)
            a = Vertex(best_f, best_t, prediction=None) #don't set prediction to value since we are splitting
            a.left_child = self._build_tree(data_l, outcome_name, curr_depth + 1)
            a.right_child = self._build_tree(data_r, outcome_name, curr_depth + 1)

            return a


    def fit(self, Xmat, Y, outcome_name="Y"):
        """
        Fit a decision tree model using training data Xmat and Y.

        Xmat: pandas dataframe of features
        Y: numpy array of 0/1 outcomes
        outcome_name: string corresponding to name of outcome variable
        """

        data = Xmat.copy()
        data[outcome_name] = Y
        self.root = self._build_tree(data, outcome_name, 0)


    def _dfs_to_leaf(self, sample):
        """
        Perform a depth first traversal to find the leaf node that the given sample belongs to

        sample: dictionary mapping from feature names to values of the feature
        """
        # vis = set()
        visited = []
        visited.append(self.root)
        while len(visited) > 0:
            pop = visited[-1]
            visited.pop()
            # if pop not in vis:
                # print("do we even use this?")
                # vis.add(pop)

            # only leaf nodes have prediction initialized
            # print("p", pop.prediction)
            if pop.prediction != None:
                return pop.prediction
            else:
                # must have child nodes since there is no prediction value
                # print("pp", pop.prediction)
                split = pop.feature_name
                # print(split)
                # print(sample)
                val = sample[split]

                if val < pop.threshold:
                    visited.append(pop.left_child)
                else:
                    visited.append(pop.right_child)

        # should never reach this becuase there is leaf with prediction at end of each tree
        assert False
        return None


    def predict(self, Xmat):
        """
        Predict 0/1 labels for a data matrix

        Xmat: pandas dataframe
        """

        predictions = []

        for i in range(len(Xmat)):

            example_i = {feature: Xmat[feature][i] for feature in Xmat.columns}
            predictions.append(self._dfs_to_leaf(example_i))

        return np.array(predictions)

    def print_tree(self, vertex=None, indent="  "):
        """
        Function to produce text representation of the tree
        """

        # initialize to root node
        if not vertex:
            vertex = self.root

        # if we're at the leaf output the prediction
        if vertex.prediction is not None:
            print("Output", vertex.prediction)

        else:
            print(vertex.feature_name, "<", round(vertex.threshold, 2), "?")
            print(indent, "Left child: ", end="")
            self.print_tree(vertex.left_child, indent + indent)
            print(indent, "Right child: ", end="")
            self.print_tree(vertex.right_child, indent + indent)


def main():
    """
    Edit only the one line marked as # EDIT ME in this function. The rest is used for grading purposes
    """


    #################
    # Simulated data
    #################
    np.random.seed(333)
    Xmat, Y  = load_simulated_data()
    data = Xmat.copy()
    data["Y"] = Y

    # test for your predict method
    # by manually creating a decision tree
    model = DecisionTree()
    model.root = Vertex(feature_name="X2", threshold=1.2)
    model.root.left_child = Vertex(prediction=0)
    model.root.right_child = Vertex(feature_name="X1", threshold=1.2)
    model.root.right_child.left_child = Vertex(prediction=0)
    model.root.right_child.right_child = Vertex(prediction=1)
    print("-"*60 + "\n" + "Hand crafted tree for testing predict\n" + "-"*60)
    model.print_tree()
    Yhat = model.predict(Xmat)
    print("Accuracy of hand crafted tree", round(accuracy(Y, Yhat), 2), "\n")

    # test for your best split method
    print("-"*60 + "\n" + "Simple test for finding best split\n" + "-"*60)
    model = DecisionTree(max_depth=2)
    best_feature, threshold, _, _ = model._get_best_split(data, "Y")
    print("Best feature and threshold found", best_feature, round(threshold, 2), "\n")


    # test for your fit method
    model.fit(Xmat, Y)
    print("-"*60 + "\n" + "Algorithmically generated tree for testing build_tree\n" + "-"*60)
    model.print_tree()
    Yhat = model.predict(data)
    print("Accuracy of algorithmically generated tree", round(accuracy(Y, Yhat), 2), "\n")

    #####################
    # Transplant data
    #####################
    Xmat, Y = load_transplant_data()

    # create a train test split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xmat, Y, test_size=0.25, random_state=0)
    Xtrain.reset_index(inplace=True, drop=True)
    Xtest.reset_index(inplace=True, drop=True)

    # find best depth using a form of cross validation/bootstrapping
    possible_depths = [1, 2, 3, 4, 5]
    best_depth = 0
    best_accuracy = 0
    for depth in possible_depths:

        accuracies = []
        for i in range(5):
            Xtrain_i, Xval, Ytrain_i, Yval = train_test_split(Xtrain, Ytrain, test_size=0.3, random_state=i)
            Xtrain_i.reset_index(inplace=True, drop=True)
            Xval.reset_index(inplace=True, drop=True)
            model = DecisionTree(max_depth=depth)
            model.fit(Xtrain_i, Ytrain_i, "survival_status")
            # print(Xval.head())
            # print(Yval)
            accuracies.append(accuracy(Yval, model.predict(Xval)))

        mean_accuracy = sum(accuracies)/len(accuracies)
        if mean_accuracy > best_accuracy:
            best_accuracy = mean_accuracy
            best_depth = depth


    print("-"*60 + "\n" + "Hyperparameter tuning on transplant data\n" + "-"*60)
    print("Best depth =", best_depth, "\n")
    model = DecisionTree(max_depth=best_depth)
    model.fit(Xtrain, Ytrain, "survival_status")
    print("-"*60 + "\n" + "Final tree for transplant data\n" + "-"*60)
    model.print_tree()
    print("Test accuracy", round(accuracy(Ytest, model.predict(Xtest)), 2), "\n")


if __name__ == "__main__":
    main()
