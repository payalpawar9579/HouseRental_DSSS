import matplotlib.pyplot as plt
import pandas as pd

# Data
df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "XGBoost"],
    "Accuracy": [0.83, 0.85, 0.86],
    "MAE": [16.12, 10.89, 10.51],
    "RMSE": [20.02, 16.83, 15.99],
    "R²": [0.66, 0.76, 0.78]
})

# Convert accuracy and R² to percentage for better visual comparison
df["Accuracy (%)"] = df["Accuracy"] * 100
df["R² (%)"] = df["R²"] * 100

# Plot each graph one at a time
metrics = [
    ("Accuracy (%)", "Model Accuracy (%)", "Accuracy (%)", 'mediumseagreen', (0, 100)),
    ("MAE", "Mean Absolute Error (MAE)", "Error ($)", 'cornflowerblue', None),
    ("RMSE", "Root Mean Squared Error (RMSE)", "Error ($)", 'orange', None),
    ("R² (%)", "R² Score (%)", "R² (%)", 'orchid', (0, 100))
]

figs = []

for col, title, ylabel, color, ylim in metrics:
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df["Model"], df[col], color=color)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.tight_layout()
    figs.append(fig)

plt.show()
