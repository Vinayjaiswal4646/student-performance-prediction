import matplotlib.pyplot as plt
import pandas as pd

results = {
    'Model': [
        'Linear Regression',
        'Polynomial Regression',
        'Decision Tree',
        'Random Forest',
        'MLP Regression'
    ],
    'MAE': [2.1564, 2.4691, 2.3160, 2.1112, 2.0843],
    'RMSE': [2.8618, 3.1259, 3.1076, 2.8572, 2.8205],
    'R2': [0.1602, -0.0020, 0.0097, 0.1628, 0.1843]
}

df = pd.DataFrame(results)

# MAE comparison
plt.figure(figsize=(9, 5))
plt.bar(df['Model'], df['MAE'])
plt.title('MAE Comparison of Regression Models')
plt.ylabel('MAE')
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(
    "outputs/figures/MAE_compare.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# RMSE comparison
plt.figure(figsize=(9, 5))
plt.bar(df['Model'], df['RMSE'])
plt.title('RMSE Comparison of Regression Models')
plt.ylabel('RMSE')
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(
    "outputs/figures/Rmse_compare.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# R2 comparison
plt.figure(figsize=(9, 5))
plt.bar(df['Model'], df['R2'])
plt.title('R2 Score Comparison of Regression Models')
plt.ylabel('R2 Score')
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig(
    "outputs/figures/R2_score_compare.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()