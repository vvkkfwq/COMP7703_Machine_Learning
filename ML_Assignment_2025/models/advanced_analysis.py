import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set font properties for better display
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Load dataset
print("Loading dataset...")
df = pd.read_csv("./data/assignTTSWING.csv")

# Check dataset size and shape
print(f"Dataset shape: {df.shape}")
print(f"Number of unique IDs: {df['id'].nunique()}")

# Analyze test mode and test stage
print("\nAnalyzing relationship between test mode and test stage...")
test_mode_stage = pd.crosstab(df["testmode"], df["teststage"])
print(test_mode_stage)

# Draw heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(test_mode_stage, annot=True, cmap="YlGnBu", fmt="d")
plt.title("Relationship Between Test Mode and Test Stage")
plt.xlabel("Test Stage")
plt.ylabel("Test Mode")
plt.savefig("./output/testmode_teststage_heatmap.png")
plt.close()

# Analyze gender and other features
print("\nAnalyzing relationship between gender and other features...")
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.boxplot(x="gender", y="a_max", data=df)
plt.title("Relationship Between Gender and Max Acceleration")

plt.subplot(2, 2, 2)
sns.boxplot(x="gender", y="g_max", data=df)
plt.title("Relationship Between Gender and Max Gyroscope Reading")

plt.subplot(2, 2, 3)
sns.boxplot(x="gender", y="a_entropy", data=df)
plt.title("Relationship Between Gender and Acceleration Entropy")

plt.subplot(2, 2, 4)
sns.boxplot(x="gender", y="g_entropy", data=df)
plt.title("Relationship Between Gender and Gyroscope Entropy")
plt.tight_layout()
plt.savefig("./output/gender_sensor_features.png")
plt.close()

# Detailed analysis of test mode and stage
# Calculate average sensor readings for each test mode
print("\nCalculating average sensor readings for each test mode...")
mode_analysis = df.groupby("testmode")[
    ["ax_mean", "ay_mean", "az_mean", "gx_mean", "gy_mean", "gz_mean", "a_max", "g_max"]
].mean()
print(mode_analysis)

# Draw radar chart to compare features across different test modes
print("\nDrawing radar chart for test mode features...")
# Prepare radar chart data
features = ["ax_mean", "ay_mean", "az_mean", "gx_mean", "gy_mean", "gz_mean"]
modes = [0, 1, 2]
mode_data = {}

for mode in modes:
    mode_data[mode] = df[df["testmode"] == mode][features].mean()

# Normalize data for radar chart
scaler = StandardScaler()
mode_data_scaled = {}
all_values = pd.concat([mode_data[mode] for mode in modes], axis=1)
all_values_scaled = scaler.fit_transform(all_values.T)

for i, mode in enumerate(modes):
    mode_data_scaled[mode] = all_values_scaled[i]

# Draw radar chart
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]  # Close the radar chart

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

for mode in modes:
    values = list(mode_data_scaled[mode])
    values += values[:1]  # Close the radar chart
    ax.plot(angles, values, linewidth=2, linestyle="solid", label=f"Test Mode {mode}")
    ax.fill(angles, values, alpha=0.1)

# Set radar chart labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], features)
ax.set_title("Sensor Reading Features Comparison Across Test Modes", fontsize=15)
plt.legend(loc="upper right")
plt.savefig("./output/testmode_radar_chart.png")
plt.close()

# Use PCA for dimensionality reduction
print("\nPerforming PCA dimensionality reduction analysis...")
# Select numerical features for PCA
numerical_features = [
    "ax_mean",
    "ay_mean",
    "az_mean",
    "gx_mean",
    "gy_mean",
    "gz_mean",
    "ax_var",
    "ay_var",
    "az_var",
    "gx_var",
    "gy_var",
    "gz_var",
    "a_max",
    "a_mean",
    "a_min",
    "g_max",
    "g_mean",
    "g_min",
    "a_kurt",
    "g_kurt",
    "a_entropy",
    "g_entropy",
]

# Ensure no missing values
X = df[numerical_features].dropna()
# Standardize data
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create DataFrame with PCA results and original classification information
pca_df = pd.DataFrame(
    {
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "testmode": df.loc[X.index, "testmode"],
        "teststage": df.loc[X.index, "teststage"],
        "gender": df.loc[X.index, "gender"],
    }
)

# Plot PCA results - colored by test mode
plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
sns.scatterplot(
    x="PC1", y="PC2", hue="testmode", data=pca_df, palette="viridis", alpha=0.5, s=10
)
plt.title("PCA - Colored by Test Mode")

# Plot PCA results - colored by test stage
plt.subplot(2, 2, 2)
sns.scatterplot(
    x="PC1", y="PC2", hue="teststage", data=pca_df, palette="viridis", alpha=0.5, s=10
)
plt.title("PCA - Colored by Test Stage")

# Plot PCA results - colored by gender
plt.subplot(2, 2, 3)
sns.scatterplot(
    x="PC1", y="PC2", hue="gender", data=pca_df, palette="viridis", alpha=0.5, s=10
)
plt.title("PCA - Colored by Gender")

# Explained variance ratio
plt.subplot(2, 2, 4)
explained_variance = pca.explained_variance_ratio_
plt.bar(range(1, 3), explained_variance)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title(
    f"PCA Explained Variance: PC1={explained_variance[0]:.2f}, PC2={explained_variance[1]:.2f}"
)

plt.tight_layout()
plt.savefig("./output/pca_analysis.png")
plt.close()

# Sample data for visualization
print("\nSampling data for visualization...")
# Random sampling to speed up processing
sample_size = min(10000, len(df))
sampled_indices = np.random.choice(df.index, size=sample_size, replace=False)
sampled_df = df.loc[sampled_indices]

# Select only 10 IDs for visualization
selected_ids = np.random.choice(sampled_df["id"].unique(), size=10, replace=False)
id_samples = sampled_df[sampled_df["id"].isin(selected_ids)]

# Plot scatter plot by ID
plt.figure(figsize=(12, 10))
sns.scatterplot(
    x="ax_mean", y="ay_mean", hue="id", data=id_samples, palette="tab10", alpha=0.7
)
plt.title("Sensor Reading Distribution by ID")
plt.xlabel("X-axis Acceleration Mean")
plt.ylabel("Y-axis Acceleration Mean")
plt.legend(title="ID", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("./output/id_sensor_readings.png")
plt.close()

print("\nAnalysis complete, all charts saved to output directory.")
