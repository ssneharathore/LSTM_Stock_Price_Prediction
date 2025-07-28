from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import numpy as np

def evaluate_model(model, X_test, y_test, scaler):
    # Predict
    predictions = model.predict(X_test)

    # Ensure predictions and y_test are 1D
    predictions = predictions.reshape(-1)
    y_test = y_test.reshape(-1)

    # Get number of features used during training
    n_features = scaler.n_features_in_

    # Prepare dummy arrays for inverse transform
    dummy_pred = np.zeros((len(predictions), n_features))
    dummy_actual = np.zeros((len(y_test), n_features))

    # Fill only the first column (assumed to be 'Close')
    dummy_pred[:, 0] = predictions
    dummy_actual[:, 0] = y_test

    # Inverse transform
    predictions_rescaled = scaler.inverse_transform(dummy_pred)[:, 0]
    y_actual_rescaled = scaler.inverse_transform(dummy_actual)[:, 0]

    # Evaluation metrics
    rmse = sqrt(mean_squared_error(y_actual_rescaled, predictions_rescaled))
    mae = mean_absolute_error(y_actual_rescaled, predictions_rescaled)

    return rmse, mae, predictions_rescaled, y_actual_rescaled
