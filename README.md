# 🛒 Walmart Sales Forecasting

![Python](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-active-brightgreen)

A machine learning project to forecast Walmart's weekly sales using historical data and external economic factors.

## 📌 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data](#-data)
- [Models](#-models)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

## ✨ Features

- **Comprehensive EDA** with time series decomposition
- **Feature engineering** with lag and rolling features
- **Multiple models** including Random Forest and XGBoost
- **Model evaluation** with RMSE metrics
- **Visualizations** of sales trends and feature importance

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/walmart-sales-forecasting.git
cd walmart-sales-forecasting
2. Create and activate virtual environment:
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
3. Install dependencies:
pip install -r requirements.txt

##  💻 Usage
Running the analysis:
# For exploratory data analysis
jupyter notebook notebooks/1_EDA.ipynb

# To run the full pipeline
python main.py

Example code:
from src.models.train_model import train_random_forest
model = train_random_forest(X_train, y_train)

## 📂 Project Structure
.
├── data/
│   ├── raw/            # Original CSV files
│   ├── processed/      # Cleaned data
│   └── outputs/        # Model predictions
├── notebooks/          # Jupyter notebooks
│   └── 1_EDA.ipynb     # Exploratory analysis
├── src/                # Source code
│   ├── data/           # Data loading
│   ├── features/       # Feature engineering
│   ├── models/         # ML models
│   └── visualization/  # Plotting functions
├── .gitignore
├── LICENSE
├── README.md           # This file
├── main.py             # Main pipeline
└── requirements.txt    # Dependencies

## 📊 Data
Dataset contains:

Weekly sales for 45 stores

External factors:

Temperature

Fuel price

CPI

Unemployment rate

Holiday indicators

Sample Data:

Store	Date	      Weekly_Sales	Holiday_Flag	Temperature 	Fuel_Price	 CPI	       Unemployment
1	    2010-05-02	 1,643,690	        0	          42.31	      2.572	    211.096	       8.106

## 🤖 Models
Implemented models:

1. Random Forest Regressor:

Handles non-linear relationships well

Robust to outliers

2. XGBoost Regressor:

Gradient boosting with regularization

Handles missing values

📈 Results

Model	                   RMSE	          Training Time	      Features Used

Random Forest	         $1,150,000	        2.1 min	                12

XGBoost                $1,080,000	        1.8 min               	12

Feature Importance:
https://visualizations/feature_importance.png
