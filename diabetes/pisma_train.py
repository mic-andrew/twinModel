import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('./diabetes.csv')

# Display the first few rows
print(data.head())

# Prepare features and target
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['liblinear', 'saga'],
    'l1_ratio': [0.5]  # Required for 'elasticnet' penalty
}

# Ensure that 'l1_ratio' is only used with 'elasticnet'
param_grid_filtered = [
    {k: v for k, v in param_grid.items() if k != 'l1_ratio'} if penalty != 'elasticnet' else param_grid
    for penalty in param_grid['penalty']
]

# Perform the GridSearchCV with conditional filtering
grid = GridSearchCV(LogisticRegression(random_state=42, class_weight='balanced'), param_grid_filtered, cv=5, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

# Best model after hyperparameter tuning
best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# Make predictions with the best model
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': abs(best_model.coef_[0])})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance')
plt.show()

# Save the best model
import joblib
joblib.dump(best_model, 'optimized_pisma_diabetes_model.joblib')
print("Model saved as 'optimized_pisma_diabetes_model.joblib'")
