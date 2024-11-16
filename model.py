# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Step 1: Load the dataset
file_path = '/home/kori/Desktop/School/2.2/Data science/Cat 2/Dataset - 2021.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("Dataset Preview:\n", data.head())

# Step 2: Data Cleaning and Preprocessing
# Checking for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Filling missing values (if any) with median
data.fillna(data.median(), inplace=True)

# Select features and target variable
# Replace 'Performance Index' with your desired target column
target = 'Performance Index'
features = [
    'Graduate Level of Student', 
    'Total Men', 
    'Total Women', 
    'Hours Studied', 
    'Previous_Scores', 
    'American Indian Alaskan Men', 
    'African American Women', 
    'Asian Men'
    # Add other features relevant for prediction
]

X = data[features]
y = data[target]

# Step 3: Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Train the Model
# Using Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Evaluate the Model
# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 7: Feature Importance
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title("Feature Importance")
plt.show()
