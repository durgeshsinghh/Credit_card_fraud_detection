# Credit Card Fraud Detection

A machine learning project for detecting credit card fraud using advanced classification algorithms with MLOps best practices. This project includes model training, evaluation, and a FastAPI-based REST API for real-time predictions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Stack](#technical-stack)
- [Model Information](#model-information)
- [API Documentation](#api-documentation)
- [Contributing](#contributing)
- [License](#license)

## Overview

Credit card fraud detection is a critical application in the financial industry. This project implements a comprehensive ML pipeline using scikit-learn to identify fraudulent transactions from legitimate ones. The model is trained on a dataset of credit card transactions and deployed as a production-ready API using FastAPI.

## Features

✅ **Data Preprocessing**: Automated data cleaning and feature engineering
✅ **Model Training**: Multiple ML algorithms with hyperparameter optimization
✅ **Model Evaluation**: Comprehensive metrics including precision, recall, F1-score, and AUC-ROC
✅ **REST API**: FastAPI-based service for real-time predictions
✅ **Docker Support**: Containerized deployment for easy scaling
✅ **MLOps Ready**: DVC integration for pipeline tracking and reproducibility
✅ **Visualization**: Exploratory data analysis and results visualization

## Installation

### Prerequisites

- Python 3.8+
- pip or conda
- Docker (optional, for containerized deployment)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/durgeshsinghh/Credit_card_fraud_detection.git
cd Credit_card_fraud_detection
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Setup

Build the Docker image:
```bash
docker build -t credit-fraud-detection .
```

Run the container:
```bash
docker run -p 8000:8000 credit-fraud-detection
```

## Usage

### Training the Model

To train the model with the default parameters:
```bash
make train
```

Or use DVC pipeline:
```bash
dvc repro
```

### Making Predictions via API

Start the API server:
```bash
python app.py
```

The API will be available at `http://localhost:8000`

#### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 0.0,
    "V1": -1.3598071336,
    "V2": -0.0727812558,
    "V3": 2.3600302592,
    "V4": 1.1918571547,
    "V5": -0.4146006884,
    "V6": -0.0637647075,
    "V7": -0.2957183432,
    "V8": 0.0377435874,
    "V9": 0.0846748653,
    "V10": 0.0797236721,
    "V11": -0.0378649918,
    "V12": -0.4351842812,
    "V13": -0.1437723486,
    "V14": -0.0570143655,
    "V15": -0.0287274921,
    "V16": -0.0631418351,
    "V17": -0.0846676645,
    "V18": 0.0846159096,
    "V19": -0.0196049110,
    "V20": 0.0764464541,
    "V21": 0.2210061497,
    "V22": 0.2215432235,
    "V23": -0.0458516081,
    "V24": -0.0183067735,
    "V25": 0.0262778938,
    "V26": -0.0152782053,
    "V27": -0.0059752841,
    "V28": -0.0029673953,
    "Amount": 149.62
  }'
```

#### Response

```json
{
  "prediction": 0
}
```

Where:
- `0` = Legitimate transaction
- `1` = Fraudulent transaction

## Project Structure

```
├── LICENSE                 # Project license
├── Makefile               # Makefile with useful commands
├── README.md              # This file
├── app.py                 # FastAPI application
├── data
│   ├── external           # Data from third party sources
│   ├── interim            # Intermediate transformed data
│   ├── processed          # Final datasets for modeling
│   └── raw                # Original immutable data
├── docs                   # Sphinx documentation
├── models                 # Trained models
├── notebooks              # Jupyter notebooks
│   └── exploration.ipynb  # EDA and analysis
├── references             # Data dictionaries and manuals
├── reports                # Generated analysis reports
│   └── figures            # Generated graphics
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup configuration
├── src                    # Source code
│   ├── data
│   │   └── make_dataset.py           # Data loading and preprocessing
│   ├── features
│   │   └── build_features.py         # Feature engineering
│   ├── models
│   │   ├── train_model.py            # Model training
│   │   └── predict_model.py          # Predictions
│   └── visualization
│       └── visualize.py              # Visualization utilities
├── dvc.yaml              # DVC pipeline configuration
├── dvc.lock              # DVC lock file
├── params.yaml           # Model parameters
├── Dockerfile            # Docker configuration
└── tox.ini              # Tox testing configuration
```

## Technical Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn |
| **Data Processing** | pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Web Framework** | FastAPI |
| **Server** | Uvicorn |
| **Model Serialization** | joblib |
| **MLOps** | DVC, DVCLive |
| **Containerization** | Docker |
| **Testing** | tox, pytest |
| **Code Quality** | flake8 |

## Model Information

### Dataset
- **Source**: Kaggle Credit Card Fraud Detection Dataset
- **Size**: ~284,807 transactions
- **Positive Class Ratio**: ~0.17% (highly imbalanced)
- **Features**: 30 (28 principal components + Time + Amount)

### Preprocessing
- Principal Component Analysis (PCA) applied to features
- StandardScaler normalization
- Handling of class imbalance

### Model Architecture
The project supports multiple classification algorithms:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)

### Performance Metrics
- Precision: Focus on minimizing false positives
- Recall: Focus on catching fraudulent transactions
- F1-Score: Balanced evaluation metric
- AUC-ROC: Overall model discriminative ability

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```
GET /
```

**Response:**
```json
{
  "message": "API is working fine"
}
```

#### 2. Prediction
```
POST /predict
```

**Request Body:**
- All 30 features required (Time, V1-V28, Amount)
- All values should be float

**Response:**
```json
{
  "prediction": 0 or 1
}
```

**Status Codes:**
- `200`: Successful prediction
- `422`: Validation error (missing or invalid fields)

## Making Contributions

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

Install development dependencies:
```bash
pip install -r dev-requirements.txt
```

Run tests:
```bash
make test
```

Run linting:
```bash
flake8 src/
```

## Useful Commands

```bash
make data          # Download and process data
make features      # Build features
make train         # Train model
make predict       # Make predictions
make visualize     # Generate visualizations
make test          # Run tests
make clean         # Clean up temporary files
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:
```
@misc{credit_fraud_detection,
  author = {Durgesh Singh},
  title = {Credit Card Fraud Detection},
  year = {2024},
  url = {https://github.com/durgeshsinghh/Credit_card_fraud_detection}
}
```

## Acknowledgments

- Dataset from Kaggle
- Built with scikit-learn and FastAPI
- Project template based on [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)

---

**Last Updated**: March 2024
**Author**: Durgesh Singh
**Repository**: https://github.com/durgeshsinghh/Credit_card_fraud_detection