import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# Set font properties for better display
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Load dataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "assignTTSWING.csv")
df = pd.read_csv(data_path)

# Use a richer set of features (refer to advanced_analysis.py)
full_features = [
    "ax_mean", "ay_mean", "az_mean",
    "gx_mean", "gy_mean", "gz_mean",
    "ax_var", "ay_var", "az_var",
    "gx_var", "gy_var", "gz_var",
    "a_max", "a_mean", "a_min",
    "g_max", "g_mean", "g_min",
    "a_kurt", "g_kurt",
    "a_entropy", "g_entropy",
]
X = df[full_features].dropna()
# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame with PCA results and testmode
pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "testmode": df.loc[X.index, "testmode"],
})

# Plot PCA results - colored by test mode
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="PC1", y="PC2", hue="testmode", data=pca_df, palette="viridis", alpha=0.5, s=10
)
plt.title("PCA (Full Feature Set) - Colored by Test Mode")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Test Mode")
plt.tight_layout()

# Save the plot
output_dir = os.path.join(project_root, "output", "visualization_analysis")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "pca_testmode_sensor_advanced_full.png"))
plt.close()

# Print explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance ratio: PC1={explained_variance[0]:.2f}, PC2={explained_variance[1]:.2f}") 