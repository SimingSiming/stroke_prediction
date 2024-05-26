import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import yaml
from io import StringIO

def preprocess_data(df, numeric_features, cat_features, drop_features):
    categories = {
        'gender': ['Female','Male'],
        'ever_married': ['No','Yes'],
        'work_type': ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'],
        'Residence_type': ['Rural', 'Urban'],
        'smoking_status': ['Unknown', 'formerly smoked', 'never smoked', 'smokes']
    }
    categories_list = [categories[feature] for feature in cat_features]
    # Define transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder(categories=categories_list, handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_features),
            ('cat', categorical_transformer, cat_features)
        ])
    
    # Split features and target
    X = df.drop(columns=drop_features)  

    # Apply transformations
    X_transformed = preprocessor.fit_transform(X)
    # Convert the transformed features back to a DataFrame
    X_transformed_df = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())

    return X_transformed_df 
