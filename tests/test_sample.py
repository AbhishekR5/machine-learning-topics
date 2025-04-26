import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Sample data for testing
@pytest.fixture
def sample_data():
    data = {
        'Energy': [0.8, 0.6, 0.7, 0.9, 0.5],
        'Valence': [0.5, 0.6, 0.7, 0.8, 0.4],
        'Danceability': [0.7, 0.8, 0.6, 0.9, 0.5],
        'Loudness': [-5, -6, -7, -4, -8],
        'Acousticness': [0.1, 0.2, 0.3, 0.4, 0.5],
        'Tempo': [120, 130, 125, 140, 110],
        'Speechiness': [0.05, 0.04, 0.06, 0.07, 0.03],
        'Liveness': [0.2, 0.3, 0.1, 0.4, 0.5],
        'Popularity': [60, 70, 65, 80, 50]
    }
    return pd.DataFrame(data)

def test_data_preprocessing(sample_data):
    # Test if the data preprocessing works correctly
    features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness', 'Tempo', 'Speechiness', 'Liveness']
    X = sample_data[features]
    y = sample_data['Popularity']

    assert not X.isnull().values.any(), "Features contain NaN values"
    assert not y.isnull().values.any(), "Target variable contains NaN values"
    assert X.shape[1] == len(features), "Feature selection is incorrect"

def test_train_test_split(sample_data):
    # Test if train-test split works correctly
    features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness', 'Tempo', 'Speechiness', 'Liveness']
    X = sample_data[features]
    y = sample_data['Popularity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    assert len(X_train) == 4, "Train set size is incorrect"
    assert len(X_test) == 1, "Test set size is incorrect"
    assert len(y_train) == 4, "Train target size is incorrect"
    assert len(y_test) == 1, "Test target size is incorrect"

def test_model_training(sample_data):
    # Test if the Random Forest model trains correctly
    features = ['Energy', 'Valence', 'Danceability', 'Loudness', 'Acousticness', 'Tempo', 'Speechiness', 'Liveness']
    X = sample_data[features]
    y = sample_data['Popularity']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    assert len(y_pred) == len(y_test), "Prediction size is incorrect"
    assert mean_squared_error(y_test, y_pred) >= 0, "MSE should be non-negative"