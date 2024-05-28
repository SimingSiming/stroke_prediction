# stroke_prediction
This is for MLDS Cloud Engineering project to predict stroke disease.

# Create the Environment File:
```bash
echo -e "AWS_ACCESS_KEY_ID=your_access_key_id\nAWS_SECRET_ACCESS_KEY=your_secret_access_key\nOTHER_ENV_VARIABLE=value" > .env
```

# Docker files:
Build the Docker Image:

```bash
docker build -t stroke_prediction:latest .
```

Build the Heart Pipeline Image:
```bash
docker build --platform linux/x86_64 -f dockerfiles/Dockerfile -t heart-pipeline .
```

Build the Heart Webapp Image:
```bash
docker build --platform linux/x86_64 -t heart-webapp .
```

Run the Heart Pipeline Container:
```bash
docker run --rm --env-file .env -v ~/.aws:/root/.aws heart-pipeline
```

Run the Heart Webapp Container:
```bash
docker run --rm ~/.aws:/root/.aws heart-webapp
```
