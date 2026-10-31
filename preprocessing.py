import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def prepare_data():

    # Load dataset
    df = pd.read_csv("data/student_por.csv", sep=";")

    # Separate features and target
    X = df.drop(columns=['G1', 'G2', 'G3'])
    y = df['G3']

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(
        include=['int64', 'float64']
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=['str', 'object']
    ).columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor