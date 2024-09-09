import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load and prepare data (as before)
data = pd.read_csv('./diabetes_prediction_dataset.csv')
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Split the data (as before)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps (as before)
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

# Create and train the model (as before)
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression(random_state=42))])
clf.fit(X_train, y_train)

# Calculate feature statistics
feature_stats = {}
for feature in numeric_features + categorical_features:
    if feature in numeric_features:
        feature_stats[feature] = {
            'mean_diabetes': data[data['diabetes'] == 1][feature].mean(),
            'mean_no_diabetes': data[data['diabetes'] == 0][feature].mean(),
            'percentile_75_diabetes': data[data['diabetes'] == 1][feature].quantile(0.75),
            'percentile_75_no_diabetes': data[data['diabetes'] == 0][feature].quantile(0.75)
        }
    else:
        feature_stats[feature] = {}
        for category in data[feature].unique():
            diabetes_count = ((data[feature] == category) & (data['diabetes'] == 1)).sum()
            total_count = (data[feature] == category).sum()
            feature_stats[feature][category] = diabetes_count / total_count if total_count > 0 else 0

# Save the model and feature statistics
joblib.dump({
    'model': clf,
    'feature_stats': feature_stats
}, 'diabetes_prediction_model.joblib')

print("Model and feature statistics saved as 'diabetes_prediction_model.joblib'")