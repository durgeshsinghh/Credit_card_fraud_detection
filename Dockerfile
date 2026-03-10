# Use official Python runtime
FROM python:3.8-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install AWS CLI (optional if pulling model from S3)
RUN pip install awscli

# Create models directory
RUN mkdir -p /app/models

# Expose FastAPI port
EXPOSE 8080

# Run FastAPI using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]