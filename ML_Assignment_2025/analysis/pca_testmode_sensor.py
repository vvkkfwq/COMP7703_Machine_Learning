import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Load the dataset
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(project_root, "data", "assignTTSWING.csv")
df = pd.read_csv(data_path)

# Select numeric sensor features (exclude non-numeric columns and 'testmode')
feature_columns = [col for col in df.columns if col != 'testmode']
X = df[feature_columns].select_dtypes(include=[np.number])
y = df['testmode']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Plot the PCA result, colored by testmode
plt.figure(figsize=(10, 8))
for mode in sorted(y.unique()):
    idx = y == mode
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=f'Test Mode {mode}', alpha=0.6)
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Sensor Features Colored by Test Mode')
plt.legend()
plt.tight_layout()

# Save the plot
output_dir = os.path.join(project_root, "output", "visualization_analysis")
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, "pca_testmode_sensor.png"))
plt.close()

# Optionally, print explained variance ratio
print("Explained variance ratio:", pca.explained_variance_ratio_) 