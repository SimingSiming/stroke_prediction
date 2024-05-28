from unittest.mock import mock_open, patch, MagicMock, Mock
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to the Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)
# Mock some external imports that we will use
import src.evaluate as e
import src.preprocess as pre
import src.train as t


def test_predict_model():
    """Test predict_model function"""
    model = MagicMock()
    X_test = pd.DataFrame(np.random.rand(10, 5))
    model.predict.return_value = np.random.randint(0, 2, size=10)
    model.predict_proba.return_value = np.random.rand(10, 2)

    y_pred, y_proba = e.predict_model(model, X_test)
    
    assert len(y_pred) == 10
    assert len(y_proba) == 10

def test_evaluate_performance():
    """Test evaluate_performance function"""
    y_test = np.random.randint(0, 2, size=10)
    y_pred = np.random.randint(0, 2, size=10)
    y_proba = np.random.rand(10)
    metrics_to_evaluate = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1_score']

    metrics = e.evaluate_performance(y_test, y_pred, y_proba, metrics_to_evaluate)
    
    for metric in metrics_to_evaluate:
        assert metric in metrics

def test_plot_roc_auc():
    """Test plot_roc_auc function"""
    y_test = np.random.randint(0, 2, size=10)
    y_proba = np.random.rand(10)
    output_path = "test_roc_auc.png"

    with patch("matplotlib.pyplot.savefig") as mock_savefig:
        e.plot_roc_auc(y_test, y_proba, output_path)
        mock_savefig.assert_called_once_with(output_path)
def test_split_data():
    """Test split_data function"""
    df = pd.DataFrame(np.random.rand(100, 5), columns=['A', 'B', 'C', 'D', 'target'])
    target = 'target'
    test_size = 0.2
    random_state = 42

    X_train, X_test, y_train, y_test = t.split_data(df, target, test_size, random_state)
    
    assert len(X_train) == 80
    assert len(X_test) == 20

def test_train_model():
    """Test train_model function"""
    X_train = pd.DataFrame(np.random.rand(80, 4))
    y_train = np.random.randint(0, 2, size=80)
    n_estimators = 100
    random_state = 42
    max_depth = 10

    model = t.train_model(X_train, y_train, n_estimators, random_state, max_depth)
    
    assert model is not None
    assert hasattr(model, "predict")

def test_save_model():
    """Test save_model function"""
    model = MagicMock()
    model_path = "test_model.pkl"

    with patch("builtins.open", mock_open()) as mock_file:
        t.save_model(model, model_path)
        mock_file.assert_called_once_with(model_path, 'wb')
        model.dump.assert_called_once()
def test_preprocess_data():
    """Test preprocess_data function"""
    df = pd.DataFrame({
        'num1': [1.0, 2.0, np.nan, 4.0],
        'cat1': ['A', 'B', 'A', 'B'],
        'drop_col': [1, 1, 1, 1],
        'target': [0, 1, 0, 1]
    })
    numeric_features = ['num1']
    cat_features = ['cat1']
    drop_features = ['drop_col']
    target_feature = 'target'

    cleaned_data = pre.preprocess_data(df, numeric_features, cat_features, drop_features, target_feature)
    
    assert not cleaned_data.isnull().values.any()
    assert 'target' in cleaned_data.columns