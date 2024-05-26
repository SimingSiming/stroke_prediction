# Base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /pipeline

# Copy the necessary directories and files into the container
COPY requirements.txt .
COPY pipeline.py .
COPY src/ src/
COPY config/ config/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run on container start
CMD ["python3", "pipeline.py", "--config", "config/config.yaml"]
