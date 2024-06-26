import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import yaml
import boto3
from io import StringIO

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

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_transformed, y)

    # Convert the transformed features back to a DataFrame
    X_resampled_df = pd.DataFrame(X_resampled, columns=preprocessor.get_feature_names_out())

    # Combine the transformed features and the target variable
    cleaned_data = pd.concat([X_resampled_df, y_resampled.reset_index(drop=True)], axis=1)

    return cleaned_data
