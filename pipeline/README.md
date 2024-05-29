# Heart Stroke Model Pipeline
## Project Summary

This pipeline aims to develop and deploy a machine learning model for a classification problem to predict if a person is safe from a heart stroke.  The pipeline automates the entire process from data acquisition to model evaluation and artifact storage to Amazon S3, making it easy for users to predict heart strokes using a trained model.

## Envrionment Variable Configuration 
To provide flexibility and allow different users to specify their own configurations on what models to use without modifying the code, we use a `.env` file. This file helps manage environment variables in a centralized and convenient manner, ensuring that sensitive information like AWS credentials and settings are kept secure and easily configurable.

#### How to Create an Environment File

1. Create a `.env` file in the root directory of your project.

2. Add the following content to the `.env` file:

   ```env
MODEL_SELECTION=random_forest
   ```

## Installation and Prerequisites
Ensure you have Docker installed on your system. You can download and install Docker from [here](https://www.docker.com/get-started/).

## Docker Instructions

To run the Docker container and execute the pipeline, use the following commands:

## Build the Docker image

```bash
docker build --platform linux/x86_64 -f dockerfiles/Dockerfile -t heart-pipeline .
```

## Run the entire model pipeline

```bash
docker run --rm --env-file .env -v ~/.aws:/root/.aws heart-pipeline
```

## Example Usage
When you run the pipeline, it will process the data, train a model, evaluate it, and save the results to an S3 bucket as configured. You should see logs indicating the progress of each step and the location of saved artifacts.


