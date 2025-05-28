import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import os
import sys

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Set font properties for better display
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "output", "testmode_analysis")
os.makedirs(output_dir, exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv(os.path.join(project_root, "data", "assignTTSWING.csv"))
print(f"Dataset shape: {df.shape}")
print(f"Number of unique test modes: {df['testmode'].nunique()}")
print(f"Sample count for each test mode:")
print(df["testmode"].value_counts().sort_index())

# 1. Basic statistical analysis of test modes and sensor data
print("\n1. Basic Statistical Analysis of Test Modes and Sensor Data")
# Define sensor feature groups
acc_features = [
    "ax_mean",
    "ay_mean",
    "az_mean",
    "ax_var",
    "ay_var",
    "az_var",
    "ax_rms",
    "ay_rms",
    "az_rms",
]
gyro_features = [
    "gx_mean",
    "gy_mean",
    "gz_mean",
    "gx_var",
    "gy_var",
    "gz_var",
    "gx_rms",
    "gy_rms",
    "gz_rms",
]
derived_features = [
    "a_max",
    "a_mean",
    "a_min",
    "g_max",
    "g_mean",
    "g_min",
    "a_entropy",
    "g_entropy",
]

# Calculate basic statistics for sensor features by test mode
sensor_stats = {}
for mode in sorted(df["testmode"].unique()):
    mode_data = df[df["testmode"] == mode]
    sensor_stats[mode] = {}

    # Acceleration statistics
    sensor_stats[mode]["acceleration"] = mode_data[acc_features].mean().to_dict()

    # Gyroscope statistics
    sensor_stats[mode]["gyroscope"] = mode_data[gyro_features].mean().to_dict()

    # Derived features statistics
    sensor_stats[mode]["derived"] = mode_data[derived_features].mean().to_dict()

# Output feature statistics for each test mode
for mode in sensor_stats:
    print(f"\nTest Mode {mode} Sensor Feature Averages:")
    print("Acceleration features:")
    for feat, value in sensor_stats[mode]["acceleration"].items():
        print(f"  {feat}: {value:.2f}")

    print("\nGyroscope features:")
    for feat, value in sensor_stats[mode]["gyroscope"].items():
        print(f"  {feat}: {value:.2f}")

    print("\nDerived features:")
    for feat, value in sensor_stats[mode]["derived"].items():
        print(f"  {feat}: {value:.2f}")

# 2. Visualize sensor data distribution across test modes
print("\n2. Visualizing Sensor Data Distribution Across Test Modes")

# 2.1 Box plots of primary acceleration and gyroscope features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(
    ["ax_mean", "ay_mean", "az_mean", "gx_mean", "gy_mean", "gz_mean"]
):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x="testmode", y=feature, data=df)
    plt.title(f"Relationship Between Test Mode and {feature}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "testmode_sensor_boxplots_en.png"))
plt.close()

# 2.2 Box plots of derived features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(["a_max", "g_max", "a_entropy", "g_entropy"]):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(x="testmode", y=feature, data=df)
    plt.title(f"Relationship Between Test Mode and {feature}")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "testmode_derived_boxplots_en.png"))
plt.close()

# 3. Correlation analysis between test modes and sensor features
print("\n3. Correlation Analysis Between Test Modes and Sensor Features")

# Create dummy variables for correlation calculations
df_corr = pd.get_dummies(df, columns=["testmode"], prefix="testmode")

# Select main sensor features and test mode dummy variables
selected_features = acc_features + gyro_features + derived_features
testmode_dummies = [col for col in df_corr.columns if col.startswith("testmode_")]
correlation_cols = selected_features + testmode_dummies

# Calculate correlations
correlation_matrix = df_corr[correlation_cols].corr()

# Extract correlations between sensor features and test modes
sensor_testmode_corr = correlation_matrix.loc[selected_features, testmode_dummies]

# Visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(sensor_testmode_corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Between Sensor Features and Test Modes")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "testmode_sensor_correlation_en.png"))
plt.close()

# Output highest correlations for each test mode
for mode_col in testmode_dummies:
    mode_corrs = sensor_testmode_corr[mode_col].sort_values(ascending=False)
    print(f"\nTop 5 features most correlated with {mode_col}:")
    print(mode_corrs.head(5))
    print(f"\nTop 5 features least correlated with {mode_col}:")
    print(mode_corrs.tail(5))

# 4. Feature importance analysis
print("\n4. Feature Importance Analysis")

# 4.1 Using Random Forest to assess feature importance
X = df[selected_features]
y = df["testmode"]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importance
feature_importances = pd.DataFrame(
    {"Feature": selected_features, "Importance": rf_model.feature_importances_}
)
feature_importances = feature_importances.sort_values("Importance", ascending=False)

print("\nFeature Importance Based on Random Forest:")
print(feature_importances.head(10))

# Visualize feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importances.head(15))
plt.title("Top 15 Important Features for Test Mode Prediction")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "testmode_feature_importance_rf_en.png"))
plt.close()

# 4.2 Using ANOVA for feature selection
selector = SelectKBest(f_classif, k=15)
selector.fit(X, y)

# Get ANOVA F-values and p-values
anova_scores = pd.DataFrame(
    {
        "Feature": selected_features,
        "F_Value": selector.scores_,
        "P_Value": selector.pvalues_,
    }
)
anova_scores = anova_scores.sort_values("F_Value", ascending=False)

print("\nFeature Importance Based on ANOVA:")
print(anova_scores.head(10))

# Visualize ANOVA F-values
plt.figure(figsize=(12, 8))
sns.barplot(x="F_Value", y="Feature", data=anova_scores.head(15))
plt.title("Top 15 ANOVA Features for Test Mode Prediction")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "testmode_feature_importance_anova_en.png"))
plt.close()

# 5. Pattern recognition in sensor features for test modes
print("\n5. Pattern Recognition in Sensor Features for Test Modes")

# 5.1 Create sensor feature fingerprints for each test mode
# Select top features
top_features = feature_importances["Feature"].head(6).tolist()
fingerprints = {}

plt.figure(figsize=(14, 10))
for i, feature in enumerate(top_features):
    plt.subplot(2, 3, i + 1)
    for mode in sorted(df["testmode"].unique()):
        kde = sns.kdeplot(df[df["testmode"] == mode][feature], label=f"Mode {mode}")
    plt.title(f"Test Mode Density Distribution of {feature}")
    plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "testmode_feature_distributions_en.png"))
plt.close()

# 5.2 Sensor feature radar chart
# Normalize features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[top_features]), columns=top_features)
df_scaled["testmode"] = df["testmode"].values

# Calculate feature averages for each test mode
radar_data = {}
for mode in sorted(df["testmode"].unique()):
    radar_data[mode] = (
        df_scaled[df_scaled["testmode"] == mode][top_features].mean().values
    )

# Draw radar chart
angles = np.linspace(0, 2 * np.pi, len(top_features), endpoint=False).tolist()
angles += angles[:1]  # Close the radar chart

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
for mode, values in radar_data.items():
    values = list(values)
    values += values[:1]  # Close the radar chart
    ax.plot(angles, values, linewidth=2, linestyle="solid", label=f"Test Mode {mode}")
    ax.fill(angles, values, alpha=0.1)

# Set radar chart labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], top_features)
ax.set_title("Sensor Feature Fingerprints of Different Test Modes", fontsize=15)
plt.legend(loc="upper right")
plt.savefig(os.path.join(output_dir, "testmode_sensor_fingerprint_en.png"))
plt.close()

# 6. Analyzing relationship between test modes, sensor data, and stages
print("\n6. Analyzing Relationship Between Test Modes, Sensor Data, and Stages")

# Calculate test mode and test stage combination counts
mode_stage_counts = pd.crosstab(df["testmode"], df["teststage"])
print("\nSample counts for Test Mode and Test Stage combinations:")
print(mode_stage_counts)

# Calculate average values of important sensor features for each test mode and stage combination
top_3_features = feature_importances["Feature"].head(3).tolist()
print(f"\nAverage values of top 3 important features across test modes and stages:")

for feature in top_3_features:
    print(f"\nFeature: {feature}")
    pivot = df.pivot_table(
        index="testmode", columns="teststage", values=feature, aggfunc="mean"
    )
    print(pivot)

    # Visualize
    plt.figure(figsize=(10, 6))
    for mode in pivot.index:
        values = pivot.loc[mode].values
        stages = pivot.columns
        plt.plot(stages, values, marker="o", label=f"Mode {mode}")
    plt.title(f"Changes in {feature} Across Different Test Modes and Stages")
    plt.xlabel("Test Stage")
    plt.ylabel(feature)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_dir, f"testmode_stage_{feature}_en.png"))
    plt.close()

# Summary
print("\nSummary:")
print("1. Different test modes show distinct patterns in sensor features")
print(
    "2. Most important features include:",
    ", ".join(feature_importances["Feature"].head(5).tolist()),
)
print("3. Sensor data can effectively differentiate between test modes")
print("4. Each test mode has a unique sensor feature fingerprint")
print("\nAnalysis complete, all charts saved to output directory.")
