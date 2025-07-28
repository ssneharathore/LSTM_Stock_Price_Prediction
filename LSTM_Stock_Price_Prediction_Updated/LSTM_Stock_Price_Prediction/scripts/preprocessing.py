import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_data(data, sequence_length=60):
    # Only use columns that already exist
    df = data[['Close', 'MACD', 'Signal', 'RSI']].copy()
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predicting 'Close' price

    return np.array(X), np.array(y), scaler
