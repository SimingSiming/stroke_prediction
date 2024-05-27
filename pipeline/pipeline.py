import argparse
import datetime
import logging.config
from pathlib import Path
import yaml
import pdb
import src.preprocess as pp
import src.train as t
import src.aws_utils as aws
import src.evaluate as e 
import json

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("heart_stroke")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from clouds data"
    )
    parser.add_argument(
        "--config", default="config/config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
        else:
            logger.info("Configuration file loaded from %s", args.config)

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "runs")) / str(now)
    artifacts.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Preprocess the Data
    numeric_features = config['preprocess_data']['numeric_features']
    cat_features = config['preprocess_data']['cat_features']
    drop_features = config['preprocess_data']['drop_features']
    target_feature = config['preprocess_data']['target']
    input_file_key = config['preprocess_data']['input_file']
    output_file_key = config['preprocess_data']['output_file']

    aws_config = config['aws']
    bucket_name = aws_config['bucket_name']
    region_name = aws_config['region_name']
    upload_to_s3 = aws_config['upload']
    profile_name = aws_config.get('profile_name', 'default')

    df = aws.load_data_from_s3(bucket_name, input_file_key, region_name, profile_name)
    cleaned_data = pp.preprocess_data(df, numeric_features, cat_features, drop_features, target_feature)

    if upload_to_s3:
        aws.save_data_to_s3(cleaned_data, bucket_name, output_file_key, region_name, profile_name)
        logger.info(f"File successfully uploaded to S3!! {output_file_key}")
    else:
        cleaned_data.to_csv(output_file_key, index=False)

    # Train the Model
    train_config = config['train_model']
    data_path = config['preprocess_data']['output_file']

    df = aws.load_data_from_s3(bucket_name, data_path, region_name, profile_name)

    X_train, X_test, y_train, y_test = t.split_data(df, target_feature, train_config['test_size'], train_config['random_state'])
    model = t.train_model(X_train, y_train, train_config['n_estimators'], train_config['random_state'], train_config['max_depth'])

    # Model Configuration
    model_output_path = train_config['output_model']
    model_filename = train_config['model_filename']
    s3_path = model_output_path  

    # Save and upload the model
    aws.save_and_upload_model(
        model,
        model_output_path,
        aws_config['bucket_name'],
        s3_path,
        model_filename
    )

    model_key = f"{model_output_path}{model_filename}"
    # Load the model from S3
    model = aws.load_model_from_s3(bucket_name, model_key, region_name, profile_name)
    # Now you can use the loaded model for prediction or further processing
    print("Model loaded successfully!")

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

    print(f"Metrics and ROC-AUC curve saved in {artifacts}")

    # Optionally upload artifacts to S3
    if upload_to_s3:
        aws.upload_artifacts_to_s3(artifacts, bucket_name, f"runs/{now}", region_name, profile_name)
        logger.info(f"Artifacts successfully uploaded to S3!!")








