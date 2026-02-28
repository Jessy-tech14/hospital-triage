# ============================================
# EXERCISE 2: MACHINE LEARNING TRIAGE SYSTEM
# ============================================

# STEP 1: Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


# STEP 2: Loading Dataset

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

data = pd.read_csv(
    "C:/Users/user/Downloads/heart+disease/processed.cleveland.data",
    names=column_names
)

# Cleaning dataset
data = data.replace("?", pd.NA)
data = data.dropna()
data = data.apply(pd.to_numeric)

# Converting to Binary Classification
# 0 = No Disease
# 1 = Disease (1,2,3,4 → 1)
data["target"] = data["target"].apply(lambda x: 1 if x > 0 else 0)

print("First 5 Rows:")
print(data.head())


# STEP 3: Identifying Features and Label

X = data.drop("target", axis=1)
y = data["target"]

print("\nFeatures:")
print(X.columns)
print("\nTarget values:", y.unique())


# STEP 4: Visualization

plt.scatter(data["age"], data["thalach"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.title("Age vs Maximum Heart Rate")
plt.show()


# STEP 5: Spliting Data (80% Train, 20% Test)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# STEP 6: Scaling the Data 

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# STEP 7: Training Logistic Regression Model

model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)


# STEP 8: Making Predictions

y_pred = model.predict(X_test)


# STEP 9: Evaluating Model

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)


# STEP 10: Comparing Predictions

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nComparison (First 10 Rows):")
print(comparison.head(10))


# STEP 11: Testing With New Patients 

new_patient1 = np.array([[55,1,2,140,250,0,1,150,0,2.3,0,0,2]])
new_patient1 = scaler.transform(new_patient1)
prediction1 = model.predict(new_patient1)
print("\nNew Patient 1 Prediction:", prediction1)

new_patient2 = np.array([[35,0,1,120,180,0,1,170,0,0.5,0,0,2]])
new_patient2 = scaler.transform(new_patient2)
prediction2 = model.predict(new_patient2)
print("New Patient 2 Prediction:", prediction2)