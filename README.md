# Customer Churn Data Analysis

![image](https://github.com/user-attachments/assets/5bc2e1fd-e6f9-4aca-900c-6e742e6e810d)

<!-- TABLE OF CONTENTS -->
### Table of Contents

1. [Project Overview](#Project-Overview)
2. [Getting Started](#Getting-Started)
3. [Prerequisites](#Prerequisites)
4. [Utilized Python Libraries](#Utilized-Python-Libraries)
5. [Installation](#Installation)
6. [Summary of Results and Methods Used](#Summary-of-Results-and-Methods-Used)

<!-- Project Overview -->
## About The Project

IBM has published a fictitious dataset titled Telco Customer Churn. This dataset contains information about the 7043 customers of a fictional telecommunications company that provides home phone and Internet Service. The goal is to predict customer churn. 

The company provides services such as: 
- Phone Service
- Internet Service
- Online Security
- Multiple Lines
- Device Protection
- Technical Support.
Each customer can subscribe to any of those services.

Additionally, more detailed informations are provided about the customers, including:
- their CustomerID
- gender
- tenure
- monthly and total charges
- payment method
- contract type
- senior citizen status
- partnernship status
- dependent status
- whether they left the company.

<!-- GETTING STARTED -->
## Getting Started

All the datapoints were downloaded from [Customer-Churn-Kaggle-Dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset). 

This link leads directly to the website.

### Prerequisites

[Python-3-13-2](https://www.python.org/downloads/)

Install and Import those libraries in order to access the project.

### Utilized Python Libraries:

* [Pandas](https://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Numpy](https://numpy.org/)
* [Scikit-Learn](https://scikit-learn.org/stable/)
* [Statsmodels](https://www.statsmodels.org/stable/index.html)
* [XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
* [Kmodes](https://pypi.org/project/kmodes/)

### Installation

* Pandas
  ```sh
  import pandas as pd
  ```
* Matplotlib
  ```sh
  import matplotlib.pyplot as plt
  ```
* Numpy
  ```sh
  import numpy as np
  ```
* Scikit-Learn
  ```sh
  from sklearn.decomposition import PCA
  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LogisticRegression
  from sklearn.neighbors import KNeighborsClassifier
  from sklearn.preprocessing import StandardScaler
  from sklearn.tree import DecisionTreeClassifier
  from mlxtend.feature_selection import SequentialFeatureSelector
  from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
  from sklearn.metrics import accuracy_score
  from sklearn.model_selection import learning_curve
  from sklearn.model_selection import ShuffleSplit
  from sklearn.pipeline import Pipeline
  from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut
  from sklearn.svm import SVC
  from sklearn.metrics import roc_curve, roc_auc_score, auc
  from sklearn.feature_selection import RFE
  from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
  ```
* Statsmodels
  ```sh
  from statsmodels.discrete.discrete_model import Logit
  from statsmodels.regression.linear_model import OLS
  ```
* XGBClassifier
  ```sh
  from xgboost import XGBClassifier
  ```
* KModes
  ```sh
  from kmodes.kmodes import KModes
  ```

    <p align="right">(<a href="#readme-top">back to top</a>)</p>

## Summary of Results and Methods Used

After performing customer segmentation to identify different customer types and build detailed profiles for a better understanding of their behaviors, I conducted feature selection to determine the most relevant variables for the final model. I applied logistic regression, decision trees, and random forest to compare their performance and accuracy scores across different subgroups. 

The final model included the following variables, with logistic regression achieving the highest accuracy score of 0.79.
- MultipleLines_No phone service
- InternetService_Fiber optic
- OnlineSecurity_Yes
- TechSupport_No internet service
- StreamingTV_No internet service
- StreamingMovies_Yes
- Contract_One year
- Contract_Two year
- CustomerType_longtermcustomers
- CustomerType_midtermcustomers
- CustomerType_newcustomers
- CustomerType_recently_onboarded_customers
- CustomerType_verylongtermcustomers.
    
