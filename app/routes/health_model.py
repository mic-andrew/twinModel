from tensorflow import keras
import joblib

_model = None
_scaler = None
_predictor = None

def get_model():
    global _model
    if _model is None:
        _model = keras.models.load_model('health_model.h5')
    return _model

def get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load('health_scaler.joblib')
    return _scaler

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = joblib.load('health_predictor.joblib')
    return _predictor

def save_model(model):
    model.save('health_model.h5')
    global _model
    _model = model