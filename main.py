
# ==========================================
# STUDENT PERFORMANCE PREDICTION
# STEP 1: IMPORT LIBRARIES
# ==========================================

# Data manipulation
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Visualization settings
sns.set_theme(style="whitegrid")

print("All libraries imported successfully!")

# ==========================================
# STEP 2: LOAD DATASET
# ==========================================

# Load Portuguese student performance dataset
df = pd.read_csv("C:/Users/VINAY/Desktop/project_prepdictive_anyalysis/data/student_por.csv", sep=";")

# Display first 5 records
print("First five records:")
print(df.head())

# STEP 3: DATASET INFORMATION

# Dataset dimensions
print("Dataset Shape:", df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
df.info()

# Statistical summary
print("\nStatistical Summary:")
print(df.describe(include="all"))

# STEP 4: DATA QUALITY CHECK

# Missing values
print("Missing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Data types
print("\nData Types:")
print(df.dtypes.value_counts())

# STEP 5: TARGET VARIABLE ANALYSIS

print("Final Grade Statistics:")
print(df['G3'].describe())

print("\nUnique Grades:")
print(sorted(df['G3'].unique()))

print("\nGrade Frequency:")
print(df['G3'].value_counts().sort_index())


# ==========================================
# STEP 6: FEATURE TYPE ANALYSIS
# ==========================================

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print("Numerical Columns:")
print(list(numerical_cols))

print("\nCategorical Columns:")
print(list(categorical_cols))

print("\nNumber of Numerical Features:", len(numerical_cols))
print("Number of Categorical Features:", len(categorical_cols))


# ==========================================
# STEP 7: FINAL GRADE DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

sns.histplot(df['G3'], bins=20, kde=True)

plt.title("Distribution of Final Student Grades")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig(
    "outputs/figures/grade_distribution.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==========================================
# STEP 8: GENDER DISTRIBUTION
# ==========================================

plt.figure(figsize=(6, 4))

sns.countplot(data=df, x='sex')

plt.title("Student Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Students")

plt.tight_layout()
plt.savefig(
    "outputs/figures/grade_distribution_by_gender.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print(df['sex'].value_counts())


# ==========================================
# STEP 9: STUDY TIME VS FINAL GRADE
# ==========================================

plt.figure(figsize=(8, 5))

sns.boxplot(data=df, x='studytime', y='G3')

plt.title("Study Time vs Final Grade")
plt.xlabel("Weekly Study Time")
plt.ylabel("Final Grade")

plt.tight_layout()
plt.savefig(
    "outputs/figures/study_time_vs_final_grade.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==========================================
# STEP 10: ABSENCES VS FINAL GRADE
# ==========================================

plt.figure(figsize=(8, 5))

sns.scatterplot(data=df, x='absences', y='G3')

plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")

plt.tight_layout()
plt.savefig(
    "outputs/figures/absensce_vs_final_grade.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# ==========================================
# STEP 11: CORRELATION HEATMAP
# ==========================================

# Select numerical columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Calculate correlation
correlation = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title("Correlation Heatmap of Numerical Features")

plt.tight_layout()
plt.savefig(
    "outputs/figures/heat_map.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Display correlation with target G3
print("\nCorrelation with Final Grade (G3):")
print(correlation['G3'].sort_values(ascending=False))

# Target variable
target = 'G3'

# Remove previous grades to avoid using them as predictors
X = df.drop(columns=['G1', 'G2', 'G3'])
y = df[target]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.head())

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['str', 'object']).columns

print("Numerical features:", len(numerical_features))
print("Categorical features:", len(categorical_features))

print("\nNumerical columns:")
print(list(numerical_features))

print("\nCategorical columns:")
print(list(categorical_features))

# Preprocessing pipeline

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

print("Preprocessor created successfully!")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)

print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)


0