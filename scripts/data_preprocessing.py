# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# %matplotlib inline

df=pd.read_csv('../data/BostonHousing.csv')
print(" First 5 rows of the dataset:")
df.head()

df.rename(columns={"medv": "price"}, inplace=True)
df.head()

# Create box plots
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7,h_pad=5.0)

# Check for missing values
print("Checking missing values in the dataset:")
missing_values = df.isnull().sum()
print(missing_values)

"""##Analysing the Outliers"""

import numpy as np

# Using IQR to detect outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Detecting outliers
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
print(outliers.sum())  # Count of outliers in each feature

"""Log Transformation for Highly Skewed Features"""

for col in ['crim', 'zn', 'dis', 'lstat', 'price']:
    df[col] = np.log1p(df[col])

"""Capping top and bottom of values"""

from scipy.stats.mstats import winsorize

df['rm'] = winsorize(df['rm'], limits=[0.01, 0.01])  # Cap top & bottom 1%
df['ptratio'] = winsorize(df['ptratio'], limits=[0.05, 0.05])  # Cap top & bottom 5%
df['b'] = winsorize(df['b'], limits=[0.05, 0.05])  # Cap extreme values

"""Scale using the median and interquartile range (IQR)"""

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

"""Plot after the reduction of outliers"""

# Create box plots
fig, ax = plt.subplots(ncols=7, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()

for col, value in df.items():
    sns.boxplot(y=col, data=df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7,h_pad=5.0)

"""Number of outlier in each column"""

import numpy as np

# Using IQR to detect outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Detecting outliers
outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
print(outliers.sum())  # Count of outliers in each feature

df.info()

"""###Normalize/Standardize Numerical Features"""

# Selecting numerical features for normalization
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

print("\n Standardizing numerical features...")
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

"""###Split Data into Training and Testing Sets"""

# Separate features (X) and target (y)
X = df.drop(columns=['price'])
y = df['price']

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Data Split (Training and Testing):")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

