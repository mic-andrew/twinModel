import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Step 1: Load the data
print("Step 1: Loading the data")
data = pd.read_csv('./diabetes_prediction_dataset.csv')  # Replace with your actual file path
print(data.head())
print("\nDataset shape:", data.shape)

# Step 2: Explore the data
print("\nStep 2: Exploring the data")
print(data.info())
print("\nMissing values:\n", data.isnull().sum())
print("\nTarget variable distribution:\n", data['diabetes'].value_counts(normalize=True))

# Step 3: Prepare features and target
print("\nStep 3: Preparing features and target")
X = data.drop('diabetes', axis=1)
y = data['diabetes']
print("Features:", list(X.columns))
print("Target: diabetes")

# Step 4: Split the data
print("\nStep 4: Splitting the data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Step 5: Define preprocessing steps
print("\nStep 5: Defining preprocessing steps")
numeric_features = ['age', 'hypertension', 'heart_disease', 'HbA1c_level', 'blood_glucose_level']
categorical_features = ['gender', 'smoking_history']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Step 6: Create and train the model
print("\nStep 6: Creating and training the model")
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(random_state=42))])
clf.fit(X_train, y_train)
print("Model training completed")

# Step 7: Make predictions
print("\nStep 7: Making predictions")
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Step 8: Evaluate the model
print("\nStep 8: Evaluating the model")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Step 12: Save the model
print("\nStep 12: Saving the model")


# Save the entire pipeline
joblib.dump(clf, 'diabetes_prediction_model.joblib')
print("Model saved as 'diabetes_prediction_model.joblib'")

# Optionally, save the feature names
# with open('feature_names.txt', 'w') as f:
#     for feature in feature_names:
#         f.write(f"{feature}\n")
print("Feature names saved as 'feature_names.txt'")

print("\nTo load and use the model later, you can use:")
print("loaded_model = joblib.load('diabetes_prediction_model.joblib')")
print("predictions = loaded_model.predict(new_data)")