import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'health_model.h5')
SCALER_PATH = os.path.join(BASE_DIR, 'health_scaler.joblib')
PREDICTOR_PATH = os.path.join(BASE_DIR, 'health_predictor.joblib')