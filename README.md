# Loan-Prediction

##### This project predicts loan approval based on various applicant features such as their credit score, loan amount, income, etc., using machine learning models. It uses various algorithms to evaluate loan approval and aims to create a highly accurate model to assist in predicting whether a loan application will be approved or denied.

## Table of Contents
* Introduction
* Libraries Used
* Data Preprocessing
* Modeling
* Evaluation
* Contributing

## Introduction

##### This project implements a loan prediction system using machine learning algorithms. The main goal is to use historical loan data, extract key features, and use classification models to predict if a loan should be approved or not based on input data from a new applicant.

##### We will use various preprocessing steps, including handling missing data, encoding categorical features, and scaling the features. Multiple machine learning classifiers are implemented, including Logistic Regression, Random Forest, Support Vector Machine (SVM), Gradient Boosting, Decision Trees, Naive Bayes, and K-Nearest Neighbors (KNN).

## Libraries Used
The following libraries are used in this project:

* pandas: Data manipulation and analysis.
* numpy: Numerical computations.
* matplotlib & seaborn: Data visualization.
* sklearn: Machine learning utilities, including preprocessing, modeling, and evaluation tools.

### Import Statements:

* import pandas as pd
* import numpy as np
* import matplotlib.pyplot as plt
* import seaborn as sns

### Preprocessing Libraries:

* from sklearn.impute import SimpleImputer 
* from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, FunctionTransformer
* from sklearn.model_selection import train_test_split

### Model Libraries:

* from sklearn.linear_model import LinearRegression, LogisticRegression
* from sklearn.svm import SVC
* from sklearn.tree import DecisionTreeClassifier
* from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
* from sklearn.neighbors import KNeighborsClassifier
* from sklearn.naive_bayes import CategoricalNB, GaussianNB

### Evaluation Libraries:

* from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
* from sklearn.model_selection import cross_val_score, RandomizedSearchCV

## Data Preprocessing
Before applying machine learning algorithms, data preprocessing is crucial:

1. Missing Data Handling: We use SimpleImputer to handle any missing values in the dataset.
2. Categorical Data Encoding: We use LabelEncoder and OneHotEncoder to encode categorical variables into numerical values.
3. Feature Scaling: Features are scaled using MinMaxScaler to ensure the model performs optimally.
4. Train-Test Split: The dataset is split into training and testing sets using train_test_split.

## Modeling
The following machine learning models are implemented for loan approval prediction:

* Logistic Regression: Used for binary classification tasks.
* Random Forest Classifier: A powerful ensemble method.
* Support Vector Machine (SVM): A classification model suited for both linear and non-linear classification.
* Decision Tree Classifier: A tree-based classifier.
* Gradient Boosting Classifier: A boosting model for improving model performance.
* K-Nearest Neighbors (KNN): A non-parametric method for classification.
* Naive Bayes Classifier: Based on Bayes' theorem, it handles categorical features well.

## Evaluation
Each model's performance is evaluated using:

* Accuracy Score: The overall correctness of the model.
* Confusion Matrix: A matrix showing the performance of the classification model.
* Classification Report: Provides metrics like precision, recall, and F1-score.
* Additionally, cross-validation (cross_val_score) is used to validate model performance across multiple splits of the dataset.

## Contributing
#### Feel free to contribute to this project by forking the repository and submitting pull requests. If you have ideas for improvements or encounter any issues, please open an issue on the GitHub repository.
