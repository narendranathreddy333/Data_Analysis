# Data Analysis Project

This repository contains a **Data Analysis** project that involves tidying, testing, and analyzing data using various machine learning models and visualization techniques.

## 📌 Project Overview

The primary objective of this project is to:

- **Check the data**: Perform initial exploratory data analysis.
- **Tidy the data**: Clean and preprocess the dataset.
- **Test the data**: Implement machine learning models for classification.

## 📂 Project Structure

- `Data_Analysis.ipynb` – Jupyter Notebook containing data analysis and model implementation.
- `data/` – (If applicable) Folder containing datasets used in the analysis.
- `README.md` – Documentation for the project.

## 🛠️ Technologies & Libraries Used

This project utilizes the following Python libraries:

```python
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.tree as tree
from io import StringIO
```


## 📊 Analysis Performed

- **Exploratory Data Analysis (EDA):** Checking and visualizing data distributions.
- **Data Cleaning:** Handling missing values, feature engineering.
- **Model Training & Evaluation:** Decision Tree, Random Forest with cross-validation.
- **Hyperparameter Tuning:** Using GridSearchCV for performance optimization.

## 📈 Results & Insights

- **Best Decision Tree Model:**
  - Best score: 0.96
  - Best parameters: `{'max_depth': 3, 'max_features': 4}`

- **Best Random Forest Model:**
  - Best score: 0.9733
  - Best parameters: `{'criterion': 'entropy', 'max_depth': 5, 'max_features': 4, 'splitter': 'random'}`

- **Optimized Model Selection:**
  - Best score: 0.9667
  - Best parameters: `{'criterion': 'gini', 'max_features': 1, 'n_estimators': 25}`

## ✨ Future Enhancements

- Improve feature selection techniques.
- Test additional machine learning models.
- Implement automated data preprocessing pipelines.


