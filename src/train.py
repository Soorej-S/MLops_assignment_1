import numpy as np
from model import SimpleModel
import joblib

# Generate some random data for demonstration
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Initialize and train the model
model = SimpleModel()
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'trained_model.joblib')
print("Model trained and saved.")