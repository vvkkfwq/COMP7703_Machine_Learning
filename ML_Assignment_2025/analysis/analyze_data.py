import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set font properties for better display
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Load dataset
df = pd.read_csv("./data/assignTTSWING.csv")

# Print basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Check data types
print("\nData Types:")
print(df.dtypes)

# Statistical feature analysis
print("\nNumerical Features Statistical Summary:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe().T)

# Categorical variable statistics
print("\nCategorical Variables Statistics:")
categorical_cols = ["testmode", "teststage", "gender", "handedness", "holdRacketHanded"]
for col in categorical_cols:
    print(f"\n{col} value counts:")
    print(df[col].value_counts())
    print(f"{col} unique values:", df[col].unique())

# Basic data visualization
print("\nPreparing visualization charts...")

# Create output directory
import os

if not os.path.exists("./output"):
    os.makedirs("./output")

# Age distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["age"], kde=True)
plt.title("Age Distribution")
plt.savefig("./output/general_analysis/age_distribution.png")
plt.close()

# Play years distribution
plt.figure(figsize=(10, 6))
sns.histplot(df["playYears"], kde=True)
plt.title("Playing Years Distribution")
plt.savefig("./output/general_analysis/play_years_distribution.png")
plt.close()

# Height and weight relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="height", y="weight", hue="gender")
plt.title("Height vs Weight Relationship")
plt.savefig("./output/general_analysis/height_weight_relation.png")
plt.close()

# Acceleration and gyroscope data mean relationships
plt.figure(figsize=(12, 8))
sns.pairplot(
    df[["ax_mean", "ay_mean", "az_mean", "gx_mean", "gy_mean", "gz_mean"]],
    diag_kind="kde",
)
plt.savefig("./output/general_analysis/acceleration_gyro_pairplot.png")
plt.close()

# Correlation matrix
# Select important numerical features
selected_features = [
    "ax_mean",
    "ay_mean",
    "az_mean",
    "gx_mean",
    "gy_mean",
    "gz_mean",
    "a_max",
    "a_mean",
    "a_min",
    "g_max",
    "g_mean",
    "g_min",
]
plt.figure(figsize=(14, 12))
corr = df[selected_features].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.savefig("./output/general_analysis/correlation_matrix.png")
plt.close()

# Analyze categorical features
print("\nAnalyzing Categorical Features:")
print("\nAge Categories:")
print(df["age"].value_counts())
print("\nPlaying Years Categories:")
print(df["playYears"].value_counts())
print("\nHeight Categories:")
print(df["height"].value_counts())
print("\nWeight Categories:")
print(df["weight"].value_counts())

# Analyze test mode and test stage distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x="testmode", data=df)
plt.title("Test Mode Distribution")
plt.xlabel("Test Mode")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.countplot(x="teststage", data=df)
plt.title("Test Stage Distribution")
plt.xlabel("Test Stage")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("./output/general_analysis/test_mode_stage_distribution.png")
plt.close()

# Analyze gender and handedness relationship
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x="gender", data=df)
plt.title("Gender Distribution")
plt.xlabel("Gender (0=Female, 1=Male)")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
sns.countplot(x="handedness", data=df)
plt.title("Handedness Distribution")
plt.xlabel("Handedness (0=Left, 1=Right)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("./output/general_analysis/gender_handedness_distribution.png")
plt.close()

# Print basic information
print("\nBasic Information:")
print(f"Total records in dataset: {len(df)}")
print(f"Number of unique IDs: {df['id'].nunique()}")
print(f"Number of unique dates: {df['date'].nunique()}")
print(f"Number of test modes: {df['testmode'].nunique()}")
print(f"Number of test stages: {df['teststage'].nunique()}")

print("\nBasic analysis completed, charts saved to output directory")
