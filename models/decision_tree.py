import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing import prepare_data


# Prepare data
X_train, X_test, y_train, y_test, preprocessor = prepare_data()


# Create Decision Tree pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(
        max_depth=4,
        random_state=42
    ))
])


# Train
model.fit(X_train, y_train)


# Predict
y_pred = model.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("\nDecision Tree Regression Results")
print("--------------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
