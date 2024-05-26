"""
This module provides functions for splitting data and training a random forest classifier.

"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def split_data(df, target, test_size, random_state):
    """
    Split the dataset into train and test sets.

    Args:
        df (DataFrame): The input DataFrame containing the dataset.
        target (str): The name of the target variable.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
        tuple: A tuple containing the train-test split arrays for features and target.

    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, n_estimators, random_state, max_depth):
    """
    Train a random forest classifier.

    Args:
        X_train (array-like): The training input samples.
        y_train (array-like): The target values for the training set.
        n_estimators (int): The number of trees in the forest.
        random_state (int): Controls both the randomness of the bootstrapping of the samples used when building trees and the sampling of the features to consider when looking for the best split at each node.
        max_depth (int): The maximum depth of the tree.

    Returns:
        RandomForestClassifier: The trained random forest classifier.

    """
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model
