# Credit Card Default Prediction

## 📌 Project Overview

This project focuses on analyzing and preprocessing a credit card customer dataset for **credit card default prediction** using Python and machine learning techniques.

The dataset contains customer demographic information, credit limit, repayment status, bill amounts, payment amounts, and the target variable indicating whether the customer will default on the next month's payment.

## 🎯 Objective

The main objectives of this project are:

* To explore the credit card dataset
* To understand the data using statistical analysis and visualizations
* To check missing values and duplicate records
* To handle categorical variables
* To detect and handle outliers
* To balance the target classes
* To select important features
* To transform and scale the data
* To prepare the dataset for machine learning

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Imbalanced-learn
* Jupyter Notebook

## 📂 Dataset

The dataset contains information about credit card customers, including:

* Customer ID
* Credit Limit
* Gender
* Education
* Marital Status
* Age
* Repayment Status
* Bill Amounts
* Payment Amounts
* Default Payment Status

The target variable is:

```text
default payment next month
```

which is renamed as:

```text
target
```

## 🔄 Project Workflow

```text
Dataset Loading
       ↓
Data Exploration
       ↓
Missing Value & Duplicate Check
       ↓
Correlation Analysis
       ↓
Column Renaming
       ↓
Categorical Data Conversion
       ↓
Outlier Treatment
       ↓
Encoding
       ↓
SMOTE Class Balancing
       ↓
Feature Selection
       ↓
Power Transformation
       ↓
Feature Scaling
       ↓
Train-Test Split
```

## 🔍 Data Preprocessing

### 1. Data Exploration

The dataset is explored using Pandas to understand:

* Dataset shape
* Column names
* Data types
* First and last records
* Statistical information
* Missing values
* Duplicate values

A correlation heatmap is also used to understand relationships between numerical features.

### 2. Column Renaming

The repayment, bill, and payment columns are renamed using month names to make the dataset easier to understand.

Examples:

```text
PAY_0 → SEP_PAY
PAY_2 → AUG_PAY
PAY_3 → JUL_PAY

BILL_AMT1 → SEP_BILL
BILL_AMT2 → AUG_BILL

PAY_AMT1 → SEP_PAYMENT
```

The target column is renamed as:

```text
default payment next month → target
```

### 3. Categorical Data Conversion

Categorical values are converted into meaningful labels.

Examples:

```text
SEX:
1 → M
2 → F
```

```text
MARRIAGE:
1 → MARRIED
2 → SINGLE
3 → OTHERS
0 → OTHERS
```

```text
EDUCATION:
1 → UG
2 → PG
3 → HIGH_SCHOOL
4 → OTHERS
0, 5, 6 → OTHERS
```

The target values are converted into:

```text
1 → YES
0 → NO
```

### 4. Outlier Treatment

The **Interquartile Range (IQR)** method is used to detect and handle outliers in numerical features.

The IQR is calculated as:

```text
IQR = Q3 - Q1
```

The lower and upper limits are calculated using:

```text
Lower Limit = Q1 - 1.5 × IQR
Upper Limit = Q3 + 1.5 × IQR
```

Extreme values are replaced with the corresponding boundary values.

### 5. Encoding

Categorical variables are converted into numerical form using encoding techniques.

The project uses:

* Label Encoding
* One-Hot Encoding

These transformations make categorical information suitable for machine learning algorithms.

### 6. Class Balancing using SMOTE

The dataset may contain an imbalance between default and non-default customers.

To address this, **SMOTE (Synthetic Minority Over-sampling Technique)** is used.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE()
x_smote, y_smote = smote.fit_resample(X, y)
```

SMOTE generates synthetic samples for the minority class and helps create a more balanced dataset.

### 7. Feature Selection

The project uses **SelectKBest** with the ANOVA F-test to select the most important features.

```python
SelectKBest(score_func=f_classif, k=25)
```

The top **25 features** are selected for further processing.

### 8. Power Transformation

A **Yeo-Johnson Power Transformation** is applied to transform the selected features and improve their distribution.

### 9. Feature Scaling

The features are standardized using:

```python
StandardScaler()
```

Scaling ensures that the features are brought to a comparable scale.

### 10. Train-Test Split

The processed dataset is divided into training and testing data.

```python
test_size = 0.2
random_state = 40
```

This means:

* **80%** of the data is used for training
* **20%** of the data is used for testing

## 📊 Project Status

| Task                   | Status        |
| ---------------------- | ------------- |
| Dataset Loading        | ✅ Completed   |
| Data Exploration       | ✅ Completed   |
| Missing Value Check    | ✅ Completed   |
| Duplicate Check        | ✅ Completed   |
| Correlation Analysis   | ✅ Completed   |
| Data Cleaning          | ✅ Completed   |
| Outlier Treatment      | ✅ Completed   |
| Encoding               | ✅ Completed   |
| SMOTE Balancing        | ✅ Completed   |
| Feature Selection      | ✅ Completed   |
| Data Transformation    | ✅ Completed   |
| Feature Scaling        | ✅ Completed   |
| Train-Test Split       | ✅ Completed   |
| Machine Learning Model | ⏳ To be added |
| Model Evaluation       | ⏳ To be added |

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn jupyter
```

### 3. Open the Notebook

```bash
jupyter notebook
```

Open the project `.ipynb` file and run the cells.

## 📁 Project Structure

```text
Credit-Card-Default-Prediction/
│
├── credit_card_project.ipynb
├── creditcard.csv
└── README.md
```

## 🔮 Future Enhancements

The project can be further extended by:

* Training machine learning classification models
* Comparing different classification algorithms
* Evaluating model accuracy
* Calculating Precision, Recall and F1-score
* Creating a confusion matrix
* Performing hyperparameter tuning
* Selecting the best-performing model
* Using the trained model to predict credit card defaults

## 👩‍💻 Project Summary

This project demonstrates the complete **data preprocessing pipeline for credit card default prediction**, starting from dataset exploration and ending with a scaled train-test dataset ready for machine learning.
