# ============================================================
# WEEK 4: SUPERVISED LEARNING MODEL IMPLEMENTATION
# Iris Flower Classification using Random Forest
# ============================================================

# ------------------------------------------------------------
# 1. Import Required Libraries
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------------------------------
# 2. Load the Dataset
# ------------------------------------------------------------

iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add target column
df["target"] = iris.target

# Add flower species name
df["species"] = df["target"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nChecking missing values:")
print(df.isnull().sum())

# ------------------------------------------------------------
# 3. Basic Data Analysis
# ------------------------------------------------------------

print("\nStatistical Summary:")
print(df.describe())

print("\nNumber of samples in each species:")
print(df["species"].value_counts())

# ------------------------------------------------------------
# 4. Feature Selection
# ------------------------------------------------------------

X = df[iris.feature_names]
y = df["target"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# ------------------------------------------------------------
# 5. Train-Test Split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data size:", X_train.shape)
print("Testing data size:", X_test.shape)

# ------------------------------------------------------------
# 6. Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# ------------------------------------------------------------
# 7. Create the Machine Learning Model
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ------------------------------------------------------------
# 8. Train the Model
# ------------------------------------------------------------

model.fit(X_train_scaled, y_train)

print("\nModel training completed.")

# ------------------------------------------------------------
# 9. Make Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)

print("\nPredicted values:")
print(y_pred)

print("\nActual values:")
print(y_test.values)

# ------------------------------------------------------------
# 10. Calculate Accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

print("\nModel Accuracy Percentage:")
print(accuracy * 100, "%")

# ------------------------------------------------------------
# 11. Classification Report
# ------------------------------------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# ------------------------------------------------------------
# 12. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")

plt.show()

# ------------------------------------------------------------
# 13. Cross-Validation
# ------------------------------------------------------------

cv_scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross-Validation Scores:")
print(cv_scores)

print("\nAverage Cross-Validation Accuracy:")
print(cv_scores.mean())

print("\nCross-Validation Accuracy Percentage:")
print(cv_scores.mean() * 100, "%")

# ------------------------------------------------------------
# 14. Feature Importance
# ------------------------------------------------------------

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": iris.feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(8, 5))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Random Forest")

plt.xticks(rotation=30)
plt.tight_layout()

plt.show()

# ------------------------------------------------------------
# 15. Test the Model with New Data
# ------------------------------------------------------------

new_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

# Scale the new data
new_flower_scaled = scaler.transform(new_flower)

# Predict
prediction = model.predict(new_flower_scaled)

print("\nNew Flower Prediction:")

print(
    "Predicted Species:",
    iris.target_names[prediction[0]]
)

# ------------------------------------------------------------
# 16. Display Prediction Probability
# ------------------------------------------------------------

probability = model.predict_proba(new_flower_scaled)

print("\nPrediction Probabilities:")

for species, prob in zip(iris.target_names, probability[0]):
    print(species, ":", round(prob * 100, 2), "%")

# ------------------------------------------------------------
# 17. Final Result
# ------------------------------------------------------------

print("\n==========================================")
print("FINAL MODEL RESULT")
print("==========================================")

print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Average Cross-Validation Accuracy:",
    round(cv_scores.mean() * 100, 2),
    "%"
)

print(
    "Predicted New Flower:",
    iris.target_names[prediction[0]]
)

print("==========================================")
