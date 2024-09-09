from flask import Blueprint, request, jsonify
import numpy as np
from models.health_model import get_model, get_scaler, get_predictor

bp = Blueprint('predict', __name__, url_prefix='/predict')

@bp.route('', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array([[
            data['heart_rate'],
            data['temperature'],
            data['activity_level']
        ]])
        model = get_model()
        scaler = get_scaler()
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0][0]
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'interpretation': 'High fever risk' if prediction > 0.5 else 'Low fever risk'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/future', methods=['POST'])
def predict_future():
    try:
        data = request.json
        features = np.array([[
            data['heart_rate'],
            data['temperature'],
            data['activity_level']
        ]])
        predictor = get_predictor()
        future_health_score = predictor.predict(features)[0]
        
        return jsonify({
            'success': True,
            'future_health_score': float(future_health_score)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@bp.route('/bulk', methods=['POST'])
def bulk_predict():
    try:
        data = request.json['data']
        df = pd.DataFrame(data)
        
        features = df[['heart_rate', 'temperature', 'activity_level']].values
        model = get_model()
        scaler = get_scaler()
        scaled_features = scaler.transform(features)
        predictions = model.predict(scaled_features).flatten()
        
        results = [
            {
                'input': row,
                'prediction': float(pred),
                'interpretation': 'High fever risk' if pred > 0.5 else 'Low fever risk'
            }
            for row, pred in zip(data, predictions)
        ]
        
        return jsonify({
            'success': True,
            'predictions': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})