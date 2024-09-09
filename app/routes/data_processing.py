import pandas as pd
import numpy as np

def preprocess_data(df):
    # Remove any rows with missing values
    df = df.dropna()
    
    # Ensure timestamp is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Normalize numerical columns
    for column in ['heart_rate', 'temperature', 'activity_level']:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
    
    return df