"""
This script defines a pipeline for acquiring, cleaning, training, and evaluating a machine learning model.

"""

import argparse
import datetime
import logging.config
from pathlib import Path
import yaml
import json
import src.preprocess as pp
import src.train as t
import src.aws_utils as aws
import src.evaluate as e

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("heart_stroke")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire, clean, and create features from clouds data")
    parser.add_argument("--config", default="config/config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
            raise e
        else:
            logger.info("Configuration file loaded from %s", args.config)

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "runs")) / str(now)
    artifacts.mkdir(parents=True, exist_ok=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Preprocess the Data
    preprocess_config = config['preprocess_data']
    numeric_features = preprocess_config['numeric_features']
    cat_features = preprocess_config['cat_features']
    drop_features = preprocess_config['drop_features']
    target_feature = preprocess_config['target']
    input_file_key = preprocess_config['input_file']
    output_file_key = preprocess_config['output_file']

    df = aws.load_data_from_s3(**preprocess_config['data_loading_params'], **config['aws'])
    cleaned_data = pp.preprocess_data(df, numeric_features, cat_features, drop_features, target_feature)

    if preprocess_config.get('upload'):
        aws.save_data_to_s3(cleaned_data, **preprocess_config['data_saving_params'], **config['aws'])
        logger.info("File successfully uploaded to S3: %s", output_file_key)
    else:
        cleaned_data.to_csv(output_file_key, index=False)

    # Train the Model
    train_config = config['train_model']
    df = aws.load_data_from_s3(**train_config['data_loading_params'], **config['aws'])

    X_train, X_test, y_train, y_test = t.split_data(df, target_feature, **train_config['split_params'])
    model = t.train_model(X_train, y_train, **train_config['model_params'])

    # Save and upload the model
    model_output_path = train_config['output_model']
    model_filename = train_config['model_filename']
    aws.save_and_upload_model(
        model,
        model_output_path,
        **train_config['model_saving_params'],
        **config['aws'],
        model_filename=model_filename
    )

    model_key = f"{model_output_path}/{model_filename}"
    model = aws.load_model_from_s3(**train_config['model_loading_params'], model_key=model_key, **config['aws'])
    logger.info("Model loaded successfully!")

    # Evaluate the Model
    evaluate_config = config['evaluate_performance']['metrics']
    y_pred, y_proba = e.predict_model(model, X_test)
    metrics = e.evaluate_performance(y_test, y_pred, y_proba, evaluate_config)

    # Save metrics to JSON with indentation for readability
    metrics_path = artifacts / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)

    # Plot and save AUC-ROC curve
    roc_auc_path = artifacts / "roc_auc_curve.png"
    e.plot_roc_auc(y_test, y_proba, roc_auc_path)

    logger.info("Metrics and ROC-AUC curve saved in %s", artifacts)

    # Optionally upload artifacts to S3
    if run_config.get('upload'):
        aws.upload_artifacts_to_s3(artifacts, **run_config['upload_params'], **config['aws'])
        logger.info("Artifacts successfully uploaded to S3: %s", artifacts)