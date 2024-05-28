import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import yaml
import boto3
from io import StringIO
import logging.config

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("heart_stroke")

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

def preprocess_data(df, numeric_features, cat_features, drop_features, target_feature):
    try:
        df = df.drop(columns=drop_features)
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, cat_features)
            ]
        )
        
        X = df.drop(columns=[target_feature])
        y = df[target_feature]
        
        X_preprocessed = preprocessor.fit_transform(X)
        
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)
        
        X_resampled_df = pd.DataFrame(X_resampled, columns=preprocessor.get_feature_names_out())
        y_resampled_df = pd.DataFrame(y_resampled, columns=[target_feature])
        
        processed_df = pd.concat([X_resampled_df, y_resampled_df], axis=1)
        
        logger.info("Data preprocessed successfully")
        return processed_df
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise

def save_data_to_s3(df, bucket_name, file_key, region_name, profile_name=None):
    try:
        session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
        s3 = session.resource('s3', region_name=region_name)
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        s3.Object(bucket_name, file_key).put(Body=csv_buffer.getvalue())
        logger.info(f"Data saved successfully to s3://{bucket_name}/{file_key}")
    except Exception as e:
        logger.error(f"Error saving data to s3://{bucket_name}/{file_key}: {e}")
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

    preprocess_config = config['preprocess_data']
    aws_config = config['aws']

    bucket_name = aws_config['bucket_name']
    region_name = aws_config['region_name']
    profile_name = aws_config.get('profile_name')

    numeric_features = preprocess_config['numeric_features']
    cat_features = preprocess_config['cat_features']
    drop_features = preprocess_config['drop_features']
    target_feature = preprocess_config['target']
    input_file_key = preprocess_config['input_file']
    output_file_key = preprocess_config['output_file']

    try:
        df = load_data_from_s3(bucket_name, input_file_key, region_name, profile_name)
        processed_df = preprocess_data(df, numeric_features, cat_features, drop_features, target_feature)
        save_data_to_s3(processed_df, bucket_name, output_file_key, region_name, profile_name)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
