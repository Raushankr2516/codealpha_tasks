# 🫀 Disease Prediction from Medical Data

## 📌 Project Overview

This project focuses on predicting the possibility of heart disease based on patient medical data using **Machine Learning classification algorithms**.

The project uses the **UCI Heart Disease Dataset** and compares four different Machine Learning algorithms:

- 🔹 Logistic Regression
- 🔹 Support Vector Machine (SVM)
- 🔹 Random Forest
- 🔹 XGBoost

The models are evaluated using **Accuracy, Precision, Recall, F1 Score, and ROC-AUC**.

---

## 🎯 Objective

The main objective of this project is to develop a Machine Learning model that can predict the possibility of heart disease based on patient medical features.

---

## 📂 Dataset

**Dataset:** UCI Heart Disease Dataset

- 📌 Total Records: **303**
- 📌 Input Features: **13**
- 📌 Target: **Heart Disease / No Heart Disease**

The original target variable contains multiple disease severity levels. For this project, it was converted into binary classification:

- 🟢 `0` = No Heart Disease
- 🔴 `1` = Heart Disease

---

## 📋 Features

The dataset contains the following features:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Gender |
| `cp` | Chest Pain Type |
| `trestbps` | Resting Blood Pressure |
| `chol` | Cholesterol Level |
| `fbs` | Fasting Blood Sugar |
| `restecg` | Resting ECG Results |
| `thalach` | Maximum Heart Rate |
| `exang` | Exercise-Induced Angina |
| `oldpeak` | ST Depression |
| `slope` | Slope of ST Segment |
| `ca` | Number of Major Vessels |
| `thal` | Thalassemia |

---

## 🛠️ Technologies Used

- 🐍 Python
- ☁️ Google Colab
- 🐼 Pandas
- 🔢 NumPy
- 📊 Matplotlib
- 🤖 Scikit-learn
- 🚀 XGBoost
- 💾 Joblib

---

## 🤖 Machine Learning Algorithms

The following classification algorithms were implemented:

1. 🔹 Logistic Regression
2. 🔹 Support Vector Machine (SVM)
3. 🔹 Random Forest
4. 🔹 XGBoost

---

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

- 📥 Data Loading
- 🔍 Data Inspection
- 🧹 Data Cleaning
- 🔄 Data Type Conversion
- 🎯 Target Variable Conversion
- 📊 Exploratory Data Analysis
- ✂️ Train-Test Split
- 📏 Feature Scaling using StandardScaler

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 86.89% | 81.25% | 92.86% | 86.67% | 87.34% |
| SVM | 85.25% | 80.65% | 89.29% | 84.75% | 85.55% |
| 🏆 Random Forest | **88.52%** | **81.82%** | **96.43%** | **88.52%** | **89.12%** |
| XGBoost | 85.25% | 78.79% | 92.86% | 85.25% | 85.82% |

---

## 🏆 Best Model

**Random Forest** achieved the best overall performance among the tested models.

### 📈 Random Forest Results

- 🎯 Accuracy: **88.52%**
- 🎯 Precision: **81.82%**
- 🎯 Recall: **96.43%**
- 🎯 F1 Score: **88.52%**
- 🎯 ROC-AUC: **89.12%**

---

## 🔄 Project Workflow

```text
📂 UCI Heart Disease Dataset
            ↓
📥 Data Loading
            ↓
🧹 Data Cleaning
            ↓
📊 Exploratory Data Analysis
            ↓
⚙️ Data Preprocessing
            ↓
✂️ Train-Test Split
            ↓
📏 Feature Scaling
            ↓
🤖 Model Training
            ↓
🔬 Model Evaluation
            ↓
📊 Model Comparison
            ↓
🏆 Best Model Selection
            ↓
🫀 Heart Disease Prediction
