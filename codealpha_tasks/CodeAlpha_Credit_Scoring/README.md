# Credit Scoring Model using Machine Learning

## 📌 Project Overview

This project focuses on predicting the creditworthiness of individuals using Machine Learning classification algorithms.

The model analyzes historical financial and personal information of customers and predicts whether a customer represents a **Good Credit Risk** or **Bad Credit Risk**.

This project was developed as part of a Machine Learning project/internship task.

## 🎯 Objective

The main objective of this project is to build a machine learning classification model that can predict customer credit risk based on historical financial information.

The project aims to:

- Analyze customer financial data
- Perform data cleaning and preprocessing
- Handle missing values
- Perform exploratory data analysis (EDA)
- Train multiple classification models
- Evaluate model performance
- Identify important features affecting credit risk
- Select the best-performing model

## 📂 Dataset

The project uses the **German Credit Risk Dataset**.

### Dataset Information

- Total Records: **1000**
- Total Features: **9**
- Target Variable: **Risk**

### Features

- Age
- Sex
- Job
- Housing
- Saving accounts
- Checking account
- Credit amount
- Duration
- Purpose

### Target

The target variable is:

- `good` → Good Credit Risk
- `bad` → Bad Credit Risk

The dataset contains:

- **700 Good Risk customers**
- **300 Bad Risk customers**

## 🛠️ Technologies Used

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

## 🔄 Project Workflow

The project follows the following Machine Learning workflow:

1. Dataset Collection
2. Data Loading
3. Data Understanding
4. Data Cleaning
5. Missing Value Handling
6. Exploratory Data Analysis
7. Categorical Encoding
8. Feature Selection
9. Train-Test Split
10. Feature Scaling
11. Model Training
12. Model Evaluation
13. Feature Importance Analysis
14. ROC Curve Comparison
15. Model Comparison
16. Final Model Selection

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

### Removing Unnecessary Columns

The `Unnamed: 0` column was removed because it represented an index and did not provide useful information for prediction.

### Handling Missing Values

Missing values were present in:

- Saving accounts
- Checking account

Since these were categorical variables, missing values were replaced using the **mode** of the respective columns.

### Encoding

Categorical variables were converted into numerical values using **Label Encoding**.

### Train-Test Split

The dataset was divided into:

- Training data: **80%**
- Testing data: **20%**

Therefore:

- Training samples: **800**
- Testing samples: **200**

## 🤖 Machine Learning Models

Three classification algorithms were implemented:

### 1. Logistic Regression

Logistic Regression was used as a baseline classification model for predicting credit risk.

### 2. Decision Tree

Decision Tree was used to capture non-linear relationships between customer features and credit risk.

### 3. Random Forest

Random Forest combines multiple decision trees to improve prediction performance and reduce overfitting.

## 📊 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 66.5% | 69.7% | 92.1% | 79.4% | 64.5% |
| Decision Tree | 60.5% | 72.3% | 70.7% | 71.5% | 53.7% |
| Random Forest | **72.0%** | **74.7%** | **90.7%** | **81.9%** | **66.9%** |

## 🏆 Best Model

Based on the evaluation results, **Random Forest** performed the best among the three tested models.

### Random Forest Performance

- Accuracy: **72.0%**
- Precision: **74.7%**
- Recall: **90.7%**
- F1-Score: **81.9%**
- ROC-AUC: **66.9%**

Therefore, Random Forest was selected as the final model for this project.

## 🔍 Feature Importance

The Random Forest model identified the following features as the most important:

| Feature | Importance |
|---|---:|
| Credit amount | 0.3028 |
| Age | 0.2094 |
| Duration | 0.1731 |
| Purpose | 0.0970 |
| Job | 0.0609 |
| Checking account | 0.0446 |
| Housing | 0.0423 |
| Saving accounts | 0.0384 |
| Sex | 0.0315 |

The analysis shows that **Credit amount, Age, and Duration** were the most influential features in the Random Forest model.

## 📁 Project Files

```text
CodeAlpha_Credit_Scoring/
│
├── Credit_Scoring_Model.ipynb
├── credit_scoring_model.pkl
├── scaler.pkl
├── README.md
└── requirements.txt
