import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv('s3://cloud-engineer-bucket/train_data.csv')

# Data Preprocessing
numerical_columns = ['age', 'avg_glucose_level', 'bmi']
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

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
        ('num', numerical_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ])

# Split
X = df.drop(columns=['id', 'stroke', 'heart_disease'])  # Drop unnecessary columns
y = df['heart_disease']  # Set target variable

X_transformed = preprocessor.fit_transform(X)

# Convert the transformed features back to a DataFrame
X_transformed_df = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())

# Combine the transformed features and the target variable
cleaned_train = pd.concat([X_transformed_df, y.reset_index(drop=True)], axis=1)

# Save cleaned data back to S3
cleaned_train.to_csv('s3://cloud-engineer-bucket/cleaned_train_data.csv', index=False)
