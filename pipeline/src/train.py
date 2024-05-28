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
import logging.config

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("heart_stroke")

def split_data(df, target, test_size, random_state):
    try:
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        logger.info("Data split into train and test sets successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

def train_model(X_train, y_train, train_config, model_selection):
    try:
        model_params = train_config[model_selection]
        if model_selection == "random_forest":
            model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', None),
                random_state=model_params.get('random_state', 42)
            )
        elif model_selection == "XGB_model":
            model = XGBClassifier(
                n_estimators=model_params.get('n_estimators', 100),
                max_depth=model_params.get('max_depth', 3),
                learning_rate=model_params.get('learning_rate', 0.1),
                random_state=model_params.get('random_state', 42)
            )
        elif model_selection == "logistic_regression":
            model = LogisticRegression(
                C=model_params.get('C', 1.0),
                max_iter=model_params.get('max_iter', 100),
                random_state=model_params.get('random_state', 42)
            )
        else:
            logger.error(f"Unsupported model selection: {model_selection}")
            raise ValueError(f"Unsupported model selection: {model_selection}")

        model.fit(X_train, y_train)
        logger.info(f"{model_selection} model trained successfully")
        return model
    except Exception as e:
        logger.error(f"Error training model {model_selection}: {e}")
        raise

def save_model(model, model_path):
    try:
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {e}")
        raise

def load_data_from_s3(bucket_name, file_key, region_name, profile_name=None):
    try:
        session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
        s3 = session.resource('s3', region_name=region_name)
        obj = s3.Object(bucket_name, file_key)
        data = obj.get()['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(data))
        logger.info(f"Data loaded successfully from s3://{bucket_name}/{file_key}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from s3://{bucket_name}/{file_key}: {e}")
        raise

if __name__ == "__main__":
    config_path = "config/config.yaml"

    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logger.info(f"Configuration file loaded from {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration file from {config_path}: {e}")
        raise

    train_config = config['train_model']
    aws_config = config['aws']

    bucket_name = aws_config['bucket_name']
    region_name = aws_config['region_name']
    profile_name = aws_config.get('profile_name')

    data_path = config['preprocess_data']['output_file']
    target_feature = config['preprocess_data']['target']
    model_output_path = train_config['output_model']
    model_selection = os.getenv('MODEL_SELECTION', 'random_forest')

    try:
        df = load_data_from_s3(bucket_name, data_path, region_name, profile_name)
        X_train, X_test, y_train, y_test = split_data(df, target_feature, train_config['test_size'], train_config['random_state'])
        model = train_model(X_train, y_train, train_config, model_selection)
        save_model(model, model_output_path)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
