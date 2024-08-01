import numpy as np
import joblib

def test_model_prediction():
    X = np.random.rand(10, 5)
   
    # Load the model from the joblib file
    loaded_model = joblib.load('trained_model.joblib')
    
    # Make predictions using the loaded model
    predictions = loaded_model.predict(X)
    
    assert len(predictions) == 10
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)