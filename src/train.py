import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib


# Generate some random data for demonstration
X = np.random.rand(100, 5)
y = np.random.randint(0, 2, 100)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'src/trained_model.joblib')
print("Model trained and saved.")
