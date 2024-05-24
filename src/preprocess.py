import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml
import boto3
from io import StringIO

def load_data_from_s3(bucket_name, file_key, region_name, profile_name):
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    s3_client = session.client('s3')
    csv_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    body = csv_obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(body))
    return df

def preprocess_data(df, numeric_features, cat_features, drop_features, target_feature):
    # Define transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_features),
            ('cat', categorical_transformer, cat_features)
        ])

    # Split features and target
    X = df.drop(columns=drop_features + [target_feature])  
    y = df[target_feature]

    # Apply transformations
    X_transformed = preprocessor.fit_transform(X)

    # Convert the transformed features back to a DataFrame
    X_transformed_df = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())

    # Combine the transformed features and the target variable
    cleaned_train = pd.concat([X_transformed_df, y.reset_index(drop=True)], axis=1)

    return cleaned_train

def save_data_to_s3(df, bucket_name, file_key, region_name, profile_name):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    s3_client = session.client('s3')
    s3_client.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

