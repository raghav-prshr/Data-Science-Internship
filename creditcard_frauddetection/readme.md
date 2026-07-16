# 💳 Credit Card Fraud Detection

A Machine Learning project that detects fraudulent credit card transactions using the Random Forest Classifier.

The project performs Exploratory Data Analysis (EDA), data preprocessing, class balancing using SMOTE, model training, and evaluation using various performance metrics and visualizations.

---

## 📌 Project Overview

Credit card fraud has become one of the major financial security challenges. Since fraudulent transactions are extremely rare compared to genuine transactions, this project uses SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset before training a Random Forest classifier.

The trained model achieves high accuracy while maintaining good precision and recall for fraud detection.

---

## 📂 Dataset

Dataset: Credit Card Fraud Detection Dataset

The dataset contains anonymized transaction features obtained using PCA.

Features:

- Time
- Amount
- V1 to V28
- Class

Target Variable:

- 0 → Genuine Transaction
- 1 → Fraudulent Transaction

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Jupyter Notebook

---

## 🤖 Machine Learning Algorithm

Random Forest Classifier

Why Random Forest?

- Handles high-dimensional data efficiently
- Reduces overfitting
- Works well on imbalanced datasets after SMOTE
- Provides Feature Importance scores

---

## 📊 Workflow

Dataset
↓
Exploratory Data Analysis (EDA)
↓
Data Preprocessing
↓
Feature Scaling
↓
Train-Test Split
↓
SMOTE
↓
Random Forest Training
↓
Model Evaluation
↓
Visualization
↓
Model Saving

---

## 📈 Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC Score
- Feature Importance

---

## 📊 Results

Accuracy

99.94%

Precision

83.5%

Recall

82.7%

F1 Score

83.1%

ROC-AUC

97.31%

---

## 📁 Project Structure

```

CreditCard_FraudDetection/

├── data/
│ └── creditcard.csv

├── outputs/
│ ├── class_distribution.png
│ ├── confusion_matrix.png
│ ├── roc_curve.png
│ ├── feature_importance.png
│ └── random_forest_model.pkl

├── src/
│ ├── eda.py
│ ├── preprocess.py
│ ├── train_model.py
│ └── evaluate.py

├── creditcard.ipynb

├── requirements.txt

└── README.md

```

---

## 📸 Output

The project automatically generates:

- Class Distribution
- Confusion Matrix
- ROC Curve
- Feature Importance Graph
- Trained Random Forest Model

These outputs are stored inside the **outputs/** directory.

---

## 🚀 Future Improvements

- Hyperparameter Tuning
- XGBoost Implementation
- LightGBM Comparison
- Deep Learning Models
- Real-time Fraud Detection API

---

## Dataset

The dataset is too large to be stored on GitHub.

Download it from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place

creditcard.csv

inside

creditcard_frauddetection/data/