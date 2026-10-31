import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

from preprocessing import prepare_data

# Load data
X_train, X_test, y_train, y_test, preprocessor = prepare_data()

# Build MLP model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True
    ))
])

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate residuals
residuals = y_test - y_pred

print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Residual:", np.mean(residuals))

# 1. Actual vs Predicted
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.7)

# Ideal prediction line
plt.plot(
    [0, 20], [0, 20],
    linestyle='--'
)

plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades - MLP")
plt.xlim(0, 20)
plt.ylim(0, 20)
plt.tight_layout()
plt.show()

# 2. Residual Plot
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, linestyle='--')

plt.xlabel("Predicted Grades")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot - MLP")
plt.tight_layout()
plt.show()

# 3. Error Distribution
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=15, edgecolor='black')

plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Distribution of Prediction Errors - MLP")
plt.tight_layout()
plt.show()