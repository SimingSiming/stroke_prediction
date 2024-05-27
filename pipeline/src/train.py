import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import yaml
import boto3
from io import StringIO

def split_data(df, target, test_size, random_state):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, train_config, model_selection):
    lr_params = train_config[model_selection]

    if model_selection == "random_forest":
        # Random Forest model
        model = RandomForestClassifier(
            n_estimators=lr_params.get('n_estimators', 100),
            max_depth=lr_params.get('max_depth', None),
            random_state=lr_params.get('random_state', 42)
        )
        model.fit(X_train, y_train)
    
    elif model_selection == "XGB_model":
        # XGBoost model
        model = XGBClassifier(
            n_estimators=lr_params.get('n_estimators', 100),
            max_depth=lr_params.get('max_depth', 3),
            learning_rate=lr_params.get('learning_rate', 0.1),
            random_state=lr_params.get('random_state', 42)
        )
        model.fit(X_train, y_train)
    
    # Logistic Regression model
    elif model_selection == "logistic_regression":
        model = LogisticRegression(
            C=lr_params.get('C', 1.0),
            max_iter=lr_params.get('max_iter', 100),
            random_state=lr_params.get('random_state', 42)
        )
        model.fit(X_train, y_train)
    
    return model
