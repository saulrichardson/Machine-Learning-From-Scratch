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

    # separate features and outcomes
    Xmat = data.drop(["Y"], axis="columns")
    Y = np.array([outcome for outcome in data["Y"]])

    return Xmat, Y


def load_transplant_data():
    """
    Helper function for loading bone
    marrow transplant data
    """


    
    # open file and process all variable names
    all_variables = []

    with open("transplant.csv", "r") as f:
        
        for line in f:
            line = line.strip()
            
            if len(line) == 0 or line[0] == "$" or line[0] == "%" or line == "@data":
                continue
                
            if line[0] == "@":
                var_name = line.split()[-2]
                all_variables.append(var_name)


    # subset data to a small number of important variables
    relevant_variables = set(["Donorage", "Recipientage", "Gendermatch",
                              "ABOmatch", "Stemcellsource", "CMVstatus", "Disease",
                              "HLAmismatch", "CD34kgx10d6", "survival_status"])
    
    data = pd.read_csv("transplant.csv", skiprows=106, names=all_variables)
    data.drop(columns=set(data.columns) - relevant_variables, inplace=True) 

    # convert disease type to numeric values
    disease_map = {disease: i for i, disease in enumerate(set(data["Disease"]))}
    data["Disease"] = [disease_map[d] for d in data["Disease"]]

    # ignore missing data
    data.replace({'?': np.nan}, regex=False, inplace=True)
    data.dropna(inplace=True)

    # convert rest of features to numeric values
    for feature in relevant_variables - set(["CD34kgx10d6", "survival_status"]):
        data[feature] = [int(value) for value in data[feature]]

    
    Xmat = data.drop(["survival_status"], axis="columns")
    Y = np.array([survival for survival in data["survival_status"]])
    return Xmat, Y



