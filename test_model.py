import numpy as np
import joblib


def test_model_prediction():
    # Define the input data
    X = np.random.rand(10, 5)

    # Load the model from the joblib file
    loaded_model = joblib.load("src/trained_model.joblib")

    # Make predictions using the loaded model
    predictions = loaded_model.predict(X)

    # Print the predictions
    print('Predicted values are:', predictions)


if __name__ == "__main__":
    test_model_prediction()
