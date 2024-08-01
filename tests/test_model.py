import sys
import os
import numpy as np
from src.model import SimpleModel
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_model_prediction():
    model = SimpleModel()
    X = np.random.rand(10, 5)
    y = np.random.randint(0, 2, 10)

    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == 10
    assert all(isinstance(pred, (int, np.integer)) for pred in predictions)
