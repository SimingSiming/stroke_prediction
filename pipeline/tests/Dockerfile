Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in Docker
WORKDIR /app

# Copy the requirements file from the parent directory into the current directory
COPY requirements.txt .

# Copy other necessary files or directories from the parent directory
COPY tests/unit_tests.py .
COPY src/ ./src/
COPY config/ ./config/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Undergo pytest 
CMD ["pytest","test_modules.py"]