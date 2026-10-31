# Student Performance Prediction Using Machine Learning

A comparative study of five machine learning regression models to predict students' final academic performance using demographic, social, family, and lifestyle factors.

## 📌 Project Overview

Student academic performance is influenced by several factors, including study habits, family background, social activities, and lifestyle choices.

This project develops and compares five machine learning regression models to predict students' final grades (`G3`) using the UCI Student Performance dataset.

The main objective is to investigate how effectively different regression algorithms can predict academic performance without using previous examination grades (`G1` and `G2`).

## 🎯 Research Objectives

- Predict students' final academic grades using machine learning.
- Compare five regression algorithms using common evaluation metrics.
- Identify influential features associated with model predictions.
- Analyze prediction errors using actual-versus-predicted values and residuals.

## 📂 Dataset

**Source:** [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

- Dataset: Portuguese student performance (`student_por.csv`)
- Total records: 649
- Original attributes: 33
- Target variable: `G3` (final grade, 0–20)
- Prediction task: Regression

The features include demographic information, family background, academic habits, social activities, and lifestyle factors.

**Important:** `G1` and `G2` are excluded from the input features to investigate prediction without previous examination grades.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## 🤖 Machine Learning Models

Five regression algorithms are implemented and evaluated:

1. Multiple Linear Regression
2. Polynomial Regression
3. Decision Tree Regression
4. Random Forest Regression
5. Multi-Layer Perceptron (MLP) Regression

All models use the same training and testing split to support a fair comparison.

## ⚙️ Methodology

1. Dataset loading and initial inspection.
2. Exploratory Data Analysis (EDA).
3. Data preprocessing:
   - Numerical feature standardization.
   - Categorical feature encoding using One-Hot Encoding.
4. Train-test split (80:20).
5. Training and evaluation of five regression models.
6. Model comparison using MAE, MSE, RMSE, and R².
7. Random Forest feature importance analysis.
8. MLP actual-versus-predicted and residual analysis.

**Experiment configuration:** `random_state=42` for the train-test split.

## 📊 Model Performance

The following results were obtained on the test dataset:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Multiple Linear Regression | 2.1564 | 2.8618 | 0.1602 |
| Polynomial Regression | 2.4691 | 3.1259 | -0.0020 |
| Decision Tree Regression | 2.3160 | 3.1076 | 0.0097 |
| Random Forest Regression | 2.1112 | 2.8572 | 0.1628 |
| MLP Regression | **2.0843** | **2.8205** | **0.1843** |

### Key Findings

- MLP Regression achieved the best overall performance among the five models.
- Random Forest Regression produced the second-best results.
- Multiple Linear Regression provided a reasonable baseline.
- Polynomial Regression and Decision Tree Regression showed weak predictive performance.
- The best model achieved an R² of approximately 0.1843, indicating that substantial variation in student grades remains unexplained.

## 📈 Visualizations

The project includes visual analysis of:

- Final grade distribution.
- Grade distribution by gender.
- Study time versus final grade.
- Absences versus final grade.
- Correlation heatmap.
- Model performance comparison.
- Random Forest feature importance.
- Actual versus predicted grades.
- Prediction residuals.

Visualizations are stored in `outputs/figures/`.

## 🔍 Feature Importance Analysis

Random Forest feature importance identified previous academic failures (`failures`) as the most influential predictor, with an importance score of approximately 0.3468.

Other influential features included:

- Absences
- Workday alcohol consumption (`Dalc`)
- Father's education (`Fedu`)
- School-related categorical features

These importance scores indicate predictive contribution within the Random Forest model; they do not establish causal relationships.

## 📁 Project Structure

```text
student-performance-prediction/
│
├── data/
│   └── student_por.csv
│
├── models/
│   ├── __init__.py
│   ├── linear_regression.py
│   ├── polynomial_regression.py
│   ├── decision_tree.py
│   ├── random_forest.py
│   └── mlp_regression.py
│
├── outputs/
│   └── figures/
│
├── main.py
├── preprocessing.py
├── compare_models.py
├── feature_importance.py
├── prediction_analysis.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Vinayjaiswal4646/student-performance-prediction.git
```

### 2. Navigate to the project directory

```bash
cd student-performance-prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python main.py
```

Run individual analysis scripts when needed:

```bash
python compare_models.py
python feature_importance.py
python prediction_analysis.py
```

## ⚠️ Limitations

- The dataset contains only 649 records from two Portuguese schools.
- The model performance is modest, with the best R² score approximately 0.1843.
- Results may not generalize to students from other countries or educational systems.
- Feature importance indicates predictive relationships, not causation.
- Cross-validation and extensive hyperparameter tuning have not been performed.

## 🔮 Future Scope

- Evaluate models on larger and more diverse educational datasets.
- Explore hyperparameter optimization.
- Investigate additional feature interpretation techniques.
- Improve generalization and prediction performance.

## 👨‍💻 Author

**Vinay Jaiswal**

B.Tech Computer Science and Engineering

GitHub: [Vinayjaiswal4646](https://github.com/V