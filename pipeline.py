import argparse
import datetime
import logging.config
from pathlib import Path
import yaml
import pdb
import src.preprocess as pp
import src.train as t
import src.score as s
import src.aws_utils as aws

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

    model_filename = config['train_model']['output_model']
    
    # Save and upload the model
    aws.save_and_upload_model(
        model,
        train_config['output_model'],
        aws_config['bucket_name'],
        train_config['output_model'],
        train_config['model_filename']
    )
    
    pdb.set_trace()



    # Score the Model
    score_config = config['score_model']
    test_data_path = config['preprocess_data']['output_file']
    model_path = config['train_model']['output_model']
    output_path = score_config['output_metrics']

    s.score_model(test_data_path, model_path, output_path, target_feature)
    logger.info("Successfully evaluated the model and saved metrics to S3!")

    

    # Generate statistics and visualizations for summarizing the data; save to disk
    figures = artifacts / "figures"
    figures.mkdir()
    eda.save_figures(features, figures, data["class"])
    logger.info("Successfuly generate figures for all features!")

    # Split data into train/test set and train model based on config; save each to disk
    tmo, train, test = tm.train_model(features, data["class"], config["train_model"])
    tm.save_data(train, test, artifacts)
    logger.info("Successfully saved Both Train Data and Test Data!")
    tm.save_model(tmo, artifacts / "trained_model_object.pkl")
    logger.info("Successfully saved the trained model!")
    # Score model on test set; save scores to disk
    scores = sm.score_model(test, tmo, config["score_model"])
    sm.save_scores(scores, artifacts / "scores.csv")
    logger.info("Successfully saved the scores into scores.csv")

    # Evaluate model performance metrics; save metrics to disk
    try:
        metrics = ep.evaluate_performance(scores, config["evaluate_performance"])
        ep.save_metrics(metrics, artifacts / "metrics.yaml")
        logger.info("Successfully evaluated performance and saved metrics.")
    except ValueError as ve:
        logger.warning("ValueError occurred during performance evaluation: %s", ve)
    except FileNotFoundError as fnf:
        logger.warning("File not found during saving metrics: %s", fnf)
    except IOError as ioe:
        logger.warning("IOError occurred while saving metrics to disk: %s", ioe)
    except yaml.YAMLError as yml:
        logger.warning("YAML related error when saving metrics: %s", yml)
    except Exception as e:
        logger.error("An unexpected error occurred during evaluation or saving metrics: %s", e)
        raise

    # Upload all artifacts to S3
    aws_config = config.get("aws")
    if aws_config.get("upload", False):
        aws.upload_artifacts(artifacts, aws_config)
