from flask import Flask, request, jsonify
import pandas as pd
import joblib
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)
# Load the model and feature statistics
model_data = joblib.load('../diabetes/diabetes_prediction_model.joblib')
model = model_data['model']
feature_stats = model_data['feature_stats']

def generate_explanation(data, prediction, probability):
    reasons = []
    high_risk_features = []
    low_risk_features = []

    for feature, value in data.items():
        if feature in feature_stats:
            if feature in ['age', 'HbA1c_level', 'blood_glucose_level']:
                mean_diabetes = feature_stats[feature]['mean_diabetes']
                percentile_75_diabetes = feature_stats[feature]['percentile_75_diabetes']
                
                if value >= percentile_75_diabetes:
                    high_risk_features.append(f"Your {feature} ({value}) is higher than 75% of people with diabetes in our dataset.")
                elif value < mean_diabetes:
                    low_risk_features.append(f"Your {feature} ({value}) is lower than the average for people with diabetes in our dataset.")
            
            elif feature in ['hypertension', 'heart_disease']:
                if value == 1:
                    high_risk_features.append(f"You have {feature}, which is associated with a higher risk of diabetes.")
                else:
                    low_risk_features.append(f"You don't have {feature}, which is associated with a lower risk of diabetes.")
            
            elif feature in ['gender', 'smoking_history']:
                diabetes_rate = feature_stats[feature].get(value, 0)
                if diabetes_rate > 0.5:
                    high_risk_features.append(f"{diabetes_rate:.1%} of people with {feature} '{value}' in our dataset have diabetes.")
                else:
                    low_risk_features.append(f"Only {diabetes_rate:.1%} of people with {feature} '{value}' in our dataset have diabetes.")

    if prediction == 1:
        reasons.append(f"Based on our analysis, you have a {probability:.1f}% chance of having diabetes.")
        reasons.append("Key factors contributing to this prediction:")
        reasons.extend(high_risk_features[:3])
        if low_risk_features:
            reasons.append("\nHowever, some factors suggest a lower risk:")
            reasons.extend(low_risk_features[:2])
    else:
        reasons.append(f"Based on our analysis, you have a {100-probability:.1f}% chance of not having diabetes.")
        if high_risk_features:
            reasons.append("Although some factors suggest an increased risk:")
            reasons.extend(high_risk_features[:2])
        reasons.append("\nKey factors contributing to this prediction:")
        reasons.extend(low_risk_features[:3])

    return reasons

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data, index=[0])
        
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]
        
        explanation = generate_explanation(data, prediction[0], probability[0] * 100)
        
        response = {
            'prediction': int(prediction[0]),
            'probability': round(float(probability[0] * 100), 1),
            'message': 'Diabetes' if prediction[0] == 1 else 'No Diabetes',
            'explanation': explanation
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)