# Use official Python runtime
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install AWS CLI (needed if pulling model from S3)
RUN pip install awscli

# Create models directory
RUN mkdir -p /app/models

# (Optional) Download model from S3
# Uncomment if using S3
# RUN aws s3 cp s3://creditcard-project/models/model.joblib /app/models/model.joblib

# Expose FastAPI port
EXPOSE 8080

# Run FastAPI application
CMD ["python", "app.py"]