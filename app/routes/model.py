from flask import Blueprint, request, jsonify
import pandas as pd
from models.health_model import get_model, get_scaler, save_model

bp = Blueprint('model', __name__, url_prefix='/model')

@bp.route('/update', methods=['POST'])
def update_model():
    try:
        data = request.json['data']
        df = pd.DataFrame(data)
        
        X = df[['heart_rate', 'temperature', 'activity_level']].values
        y = df['fever'].values
        
        model = get_model()
        scaler = get_scaler()
        X_scaled = scaler.transform(X)
        
        # Update the model
        model.fit(X_scaled, y, epochs=10, batch_size=32, verbose=0)
        
        # Save the updated model
        save_model(model)
        
        return jsonify({'success': True, 'message': 'Model updated successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})