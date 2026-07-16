# Sales Prediction Using Python

## 📌 Project Overview

This project predicts product sales based on advertising expenditures across different media channels using **Linear Regression**.

The objective is to analyze how advertising budgets influence sales and build a machine learning model capable of forecasting future sales.


---

## 📊 Dataset

The dataset contains advertising budgets spent on:

* TV

* Radio

* Newspaper

Target Variable:

* Sales

### Dataset Features

| Feature   | Description                           |

| --------- | ------------------------------------- |

| TV        | Advertising budget spent on TV        |

| Radio     | Advertising budget spent on Radio     |

| Newspaper | Advertising budget spent on Newspaper |

| Sales     | Product sales                         |

Dataset Size:

* Rows: 200

* Columns: 4

---

## 🛠 Technologies Used

* Python

* Pandas

* NumPy

* Matplotlib

* Seaborn

* Scikit-Learn

* Jupyter Notebook

---

## 📈 Exploratory Data Analysis (EDA)

The following analyses were performed:

* Dataset inspection

* Missing value analysis

* Statistical summary

* Correlation analysis

* Correlation heatmap

* Scatter plots for:

  * TV vs Sales

  * Radio vs Sales

  * Newspaper vs Sales

### Key Findings

* TV advertising showed the strongest positive relationship with Sales.

* Radio advertising also contributed positively to Sales.

* Newspaper advertising had minimal influence on Sales.

---

## 🤖 Machine Learning Model

### Algorithm Used

**Linear Regression**

### Workflow

1. Load dataset

2. Perform EDA

3. Select features and target

4. Split data into training and testing sets

5. Train Linear Regression model

6. Generate predictions

7. Evaluate model performance

---

## 📊 Model Performance

### R² Score

```text

0.9059

```

The model explains approximately **90.59%** of the variance in sales, indicating excellent predictive performance.

### Feature Coefficients

| Feature   | Coefficient |

| --------- | ----------- |

| TV        | 0.0545      |

| Radio     | 0.1009      |

| Newspaper | 0.0043      |

### Interpretation

* Increasing TV advertising budget increases sales.

* Increasing Radio advertising budget increases sales.

* Newspaper advertising has very little impact on sales.

### Intercept

```text

4.7141

```

---

## Sample Prediction

Input:

| TV  | Radio | Newspaper |

| --- | ----- | --------- |

| 150 | 25    | 20        |

Predicted Sales:

```text

15.50

```

---

## 📂 Project Structure

```text

Sales_Prediction/

│

├── data/

│   └── advertising.csv

│

├── notebooks/

│   └── Sales_Prediction.ipynb

│

├── outputs/

│   ├── correlation_heatmap.png

│   ├── tv_vs_sales.png

│   ├── radio_vs_sales.png

│   ├── newspaper_vs_sales.png

│   └── actual_vs_predicted.png

│

├── src/

│   └── sales_prediction.py

│

├── requirements.txt

│

└── README.md

```

---

## 📷 Visualizations

The project includes:

* Correlation Heatmap

* TV vs Sales Scatter Plot

* Radio vs Sales Scatter Plot

* Newspaper vs Sales Scatter Plot

* Actual vs Predicted Sales Plot

---

## ✅ Conclusion

This project successfully demonstrates how machine learning can be used to predict sales based on advertising expenditures.

The Linear Regression model achieved an R² Score of **0.9059**, indicating strong predictive capability. Among all advertising channels, **TV advertising had the strongest influence on sales**, while Newspaper advertising showed minimal impact.

---
