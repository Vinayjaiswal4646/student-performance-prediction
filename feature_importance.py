import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from preprocessing import prepare_data

# Load data
X_train, X_test, y_train, y_test, preprocessor = prepare_data()

# Build the existing Random Forest model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        random_state=42
    ))
])

# Train model
model.fit(X_train, y_train)

# Get feature names after preprocessing
feature_names = model.named_steps[
    'preprocessor'
].get_feature_names_out()

# Get importance scores
importance = model.named_steps[
    'model'
].feature_importances_

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

# Sort in descending order
importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

# Display top 15 features
print("\nTop 15 Important Features")
print("--------------------------")
print(importance_df.head(15).to_string(index=False))

# Plot feature importance
top_features = importance_df.head(15).sort_values(
    by='Importance',
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features['Feature'],
    top_features['Importance']
)

plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Top 15 Feature Importance - Random Forest")

plt.tight_layout()
plt.show()