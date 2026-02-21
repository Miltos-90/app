""" Training code """

import os
import joblib


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_iris()
X = data.data # pylint: disable=no-member
y = data.target # pylint: disable=no-member

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
os.makedirs("./artifacts", exist_ok=True)
joblib.dump(model, "./artifacts/model.pkl")

print("Model saved as model.pkl")
