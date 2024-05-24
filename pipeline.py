import argparse
import datetime
import logging.config
from pathlib import Path

import yaml

import src.preprocess as pp
# import src.analysis as eda
# import src.aws_utils as aws
# import src.create_dataset as cd
# import src.evaluate_performance as ep
# import src.generate_features as gf
# import src.score_model as sm
# import src.train_model as tm

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("clouds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from clouds data"
    )
    parser.add_argument(
        "--config", default="config/default-config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
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

    # Acquire data from online repository and save to disk
    ad.acquire_data(run_config["data_source"], artifacts / "clouds.data")

    # Create structured dataset from raw data; save to disk
    data = cd.create_dataset(artifacts / "clouds.data", config["create_dataset"])
    cd.save_dataset(data, artifacts / "clouds.csv")
    logger.info("Successfully save the dataset to %s", artifacts)

    # Enrich dataset with features for model training; save to disk
    features = gf.generate_features(data, config["generate_features"])
    features.to_csv(artifacts / "features.csv", index = False)

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
