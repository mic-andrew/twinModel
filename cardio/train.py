import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Load the data
print("Step 1: Loading the Cardiovascular Disease dataset")
data = pd.read_csv('./cardio_train.csv', delimiter=';')
print(data.head())
print("\nDataset shape:", data.shape)

# Step 2: Explore the data
print("\nStep 2: Exploring the data")
print(data.info())
print("\nMissing values:\n", data.isnull().sum())
print("\nTarget variable distribution:\n", data['cardio'].value_counts(normalize=True))

# Step 3: Prepare features and target
print("\nStep 3: Preparing features and target")
X = data.drop(['id', 'cardio'], axis=1)
y = data['cardio']
print("Features:", list(X.columns))
print("Target: cardio")

# Step 4: Split the data
print("\nStep 4: Splitting the data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 5: Scale the features
print("\nStep 5: Scaling the features")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Create and train the model
print("\nStep 6: Creating and training the model")
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
print("Model training completed")

# Step 7: Make predictions
print("\nStep 7: Making predictions")
y_pred = model.predict(X_test_scaled)

# Step 8: Evaluate the model
print("\nStep 8: Evaluating the model")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 9: Confusion Matrix
print("\nStep 9: Generating Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png')
print("Confusion Matrix saved as 'confusion_matrix.png'")

# Step 10: Save the model
print("\nStep 10: Saving the model")
joblib.dump({
    'model': model,
    'scaler': scaler
}, 'cardio_prediction_model.joblib')
print("Model and scaler saved as 'cardio_prediction_model.joblib'")

print("\nTo load and use the model later, you can use:")
print("loaded_data = joblib.load('cardio_prediction_model.joblib')")
print("model = loaded_data['model']")
print("scaler = loaded_data['scaler']")
print("predictions = model.predict(scaler.transform(new_data))")
