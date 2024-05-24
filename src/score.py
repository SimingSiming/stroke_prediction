import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import boto3

def score_model(test_data_path, model_path, output_path):
    # Load preprocessed test data
    df = pd.read_csv(test_data_path)

    # Split data into features and target variable
    X_test = df.drop(columns=['heart_disease'])
    y_test = df['heart_disease']

    # Load the trained model
    model = joblib.load(model_path)

    # Make predictions
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Print evaluation metrics
    print("Test Accuracy:", accuracy)
    print("Test Precision:", precision)
    print("Test Recall:", recall)
    print("Test F1-Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)

    # Save evaluation metrics to a CSV file
    metrics = {
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1],
        'confusion_matrix': [conf_matrix.tolist()]
    }
    metrics_df = pd.DataFrame(metrics)
    metrics_filename = '/mnt/data/evaluation_metrics.csv'
    metrics_df.to_csv(metrics_filename, index=False)

    # Upload the metrics to S3
    s3 = boto3.client('s3')
    s3.upload_file(metrics_filename, output_path.split('/')[2], '/'.join(output_path.split('/')[3:]))

