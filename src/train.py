import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import yaml
import boto3
from io import StringIO

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def split_data(df, target, test_size, random_state):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, n_estimators, random_state, max_depth):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_filename):
    joblib.dump(model, model_filename)
