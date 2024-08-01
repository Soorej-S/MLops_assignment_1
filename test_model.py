import numpy as np
import joblib
import os


def test_model_prediction():
    # Define the input data
    X = np.random.rand(10, 5)

    # Load the model from the joblib file
    model_path = os.path.join("src", "trained_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    loaded_model = joblib.load(model_path)

    # Make predictions using the loaded model
    predictions = loaded_model.predict(X)

    # Print the predictions
    print('Predicted values are:', predictions)


if __name__ == "__main__":
    test_model_prediction()
