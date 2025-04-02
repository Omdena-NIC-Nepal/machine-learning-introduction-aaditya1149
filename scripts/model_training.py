# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# %matplotlib inline

df=pd.read_csv('../data/BostonHousing.csv')
print(" First 5 rows of the dataset:")
df.head()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

df.rename(columns={"medv": "price"}, inplace=True)
df.head()

for col in ['crim', 'zn', 'dis', 'lstat', 'price']:
    df[col] = np.log1p(df[col])

from scipy.stats.mstats import winsorize

df['rm'] = winsorize(df['rm'], limits=[0.01, 0.01])  # Cap top & bottom 1%
df['ptratio'] = winsorize(df['ptratio'], limits=[0.05, 0.05])  # Cap top & bottom 5%
df['b'] = winsorize(df['b'], limits=[0.05, 0.05])  # Cap extreme values

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# Selecting numerical features for normalization
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

print("\n Standardizing numerical features...")
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# -------------------- Choose Appropriate Features --------------------

# Select highly correlated features based on the EDA
X = df.drop(['price'], axis=1)
y = df["price"]

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(" Training and Testing Data Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# -------------------- Train Linear Regression Model --------------------

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, "linear_regression_model.pkl")

# Predictions on test set
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Model saved as linear_regression_model.pkl")
print("\n Model Evaluation:")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

"""### Linear Regression has no major hyperparameters to tune, but we can use Cross-Validation to check model performance."""

cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="r2")
print("\n Cross-Validation Scores:", cv_scores)
print(f"Mean CV R²: {cv_scores.mean():.2f}")

# -------------------- Plot Results --------------------

# Plot Actual vs Predicted Values
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs Predicted Home Prices")
plt.show()

