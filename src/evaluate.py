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

def predict_model(model, X_test):
    """
    Make predictions using the trained model.
    :param model: Trained model
    :param X_test: Test features
    :return: Predicted labels and probabilities
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba

def evaluate_performance(y_test, y_pred, y_proba, metrics_to_evaluate):
    """
    Evaluate the model performance using various metrics.
    :param y_test: True labels
    :param y_pred: Predicted labels
    :param y_proba: Predicted probabilities
    :param metrics_to_evaluate: List of metrics to evaluate
    :return: Dictionary of calculated metrics
    """
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
    return metrics

def plot_roc_auc(y_test, y_proba, output_path):
    """
    Plot and save AUC-ROC curve.
    :param y_test: True labels
    :param y_proba: Predicted probabilities
    :param output_path: Path to save the plot
    """
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










