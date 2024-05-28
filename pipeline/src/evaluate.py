from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc,
    classification_report
)
import boto3
import pickle
from io import BytesIO
import matplotlib.pyplot as plt
import logging.config

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("heart_stroke")

def predict_model(model, X_test):
    try:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        logger.info("Model prediction successful")
        return y_pred, y_proba
    except Exception as e:
        logger.error(f"Error in model prediction: {e}")
        raise

def evaluate_performance(y_test, y_pred, y_proba, metrics_to_evaluate):
    try:
        metrics = {}
        if 'roc_auc' in metrics_to_evaluate:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        if 'accuracy' in metrics_to_evaluate:
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
        if 'precision' in metrics_to_evaluate:
            metrics['precision'] = precision_score(y_test, y_pred)
        if 'recall' in metrics_to_evaluate:
            metrics['recall'] = recall_score(y_test, y_pred)
        if 'f1_score' in metrics_to_evaluate:
            metrics['f1_score'] = f1_score(y_test, y_pred)
        if 'confusion_matrix' in metrics_to_evaluate:
            metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        if 'classification_report' in metrics_to_evaluate:
            metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        logger.info("Model evaluation successful")
        return metrics
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

def plot_roc_auc(y_test, y_proba, output_path):
    try:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        logger.info(f"ROC-AUC curve saved to {output_path}")
    except Exception as e:
        logger.error(f"Error plotting ROC-AUC curve: {e}")
        raise

def load_model_from_s3(bucket_name, model_key, region_name, profile_name=None):
    try:
        session = boto3.Session(profile_name=profile_name) if profile_name else boto3.Session()
        s3 = session.client('s3', region_name=region_name)
        response = s3.get_object(Bucket=bucket_name, Key=model_key)
        model_str = response['Body'].read()
        model = pickle.loads(model_str)
        logger.info(f"Model loaded successfully from s3://{bucket_name}/{model_key}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from s3://{bucket_name}/{model_key}: {e}")
        raise

if __name__ == "__main__":
    # The main function would be implemented here based on the specifics of the pipeline
    pass










