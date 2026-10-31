import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    PolynomialFeatures,
    OneHotEncoder,
    StandardScaler
)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing import prepare_data


# Prepare data
X_train, X_test, y_train, y_test, _ = prepare_data()


# Identify numerical and categorical columns
numerical_features = X_train.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=['str', 'object']
).columns.tolist()


# Apply polynomial transformation only to numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(
                degree=2,
                include_bias=False
            ))
        ]), numerical_features),

        ('cat', OneHotEncoder(
            handle_unknown='ignore'
        ), categorical_features)
    ]
)


# Create model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])


# Train model
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)


print("\nPolynomial Regression Results")
print("--------------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)