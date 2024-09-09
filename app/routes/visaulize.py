from flask import Blueprint, request, jsonify
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

bp = Blueprint('visualize', __name__, url_prefix='/visualize')

@bp.route('', methods=['GET'])
def visualize():
    try:
        days = int(request.args.get('days', 30))
        
        # Generate some example data (replace this with actual data from your database)
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        heart_rates = np.random.randint(60, 100, days)
        temperatures = np.random.uniform(36.1, 37.5, days)
        
        plt.figure(figsize=(10, 6))
        plt.plot(dates, heart_rates, label='Heart Rate')
        plt.plot(dates, temperatures, label='Temperature')
        plt.title(f'Health Trends Over Past {days} Days')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        
        # Save the plot to a bytes buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        
        # Encode the image to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{image_base64}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})