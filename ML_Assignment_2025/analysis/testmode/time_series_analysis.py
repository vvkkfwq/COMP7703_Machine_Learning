import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Set font properties for better display
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "output", "testmode_analysis", "time_series")
os.makedirs(output_dir, exist_ok=True)


def perform_time_series_analysis():
    """Analyze how sensor patterns evolve over time within each test mode"""
    print("Loading dataset...")
    df = pd.read_csv(os.path.join(project_root, "data", "assignTTSWING.csv"))
    print(f"Dataset shape: {df.shape}")

    # Assuming 'teststage' represents time progression in the swing
    # If there's a timestamp column, use that instead

    # Important features based on previous analysis
    top_features = [
        "a_min",
        "a_max",
        "g_max",
        "a_entropy",
        "g_entropy",
    ]  # Adjust based on actual findings

    # 1. Create time series plots for each test mode
    print("Creating time series plots...")

    for feature in top_features:
        plt.figure(figsize=(12, 8))

        for mode in sorted(df["testmode"].unique()):
            mode_data = df[df["testmode"] == mode].sort_values("teststage")

            # Calculate mean and standard deviation per stage
            stage_means = mode_data.groupby("teststage")[feature].mean()
            stage_stds = mode_data.groupby("teststage")[feature].std()

            # Plot the time series with confidence intervals
            plt.plot(
                stage_means.index,
                stage_means.values,
                marker="o",
                linewidth=2,
                label=f"Mode {mode}",
            )

            plt.fill_between(
                stage_means.index,
                stage_means - stage_stds,
                stage_means + stage_stds,
                alpha=0.2,
            )

        plt.title(f"Time Series of {feature} Across Test Stages by Test Mode")
        plt.xlabel("Test Stage")
        plt.ylabel(feature)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"time_series_{feature}.png"))
        plt.close()

    # 2. Create heatmaps showing feature changes over stages for each test mode
    print("Creating heatmaps of feature changes over stages...")

    for mode in sorted(df["testmode"].unique()):
        # Filter data for the current test mode
        mode_data = df[df["testmode"] == mode]

        # Create a pivot table with stages as rows and features as columns
        pivot_data = pd.DataFrame()

        for feature in top_features:
            feature_means = mode_data.groupby("teststage")[feature].mean()
            # Normalize to make features comparable
            feature_means = (feature_means - feature_means.min()) / (
                feature_means.max() - feature_means.min()
            )
            pivot_data[feature] = feature_means

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, cmap="viridis", annot=True, fmt=".2f")
        plt.title(f"Test Mode {mode}: Feature Changes Across Stages")
        plt.ylabel("Test Stage")
        plt.savefig(os.path.join(output_dir, f"heatmap_mode_{mode}.png"))
        plt.close()

    # 3. Analyze rate of change between stages
    print("Analyzing rate of change between stages...")

    # Create a dataframe to store rate of change information
    rate_of_change_df = pd.DataFrame()

    for mode in sorted(df["testmode"].unique()):
        for feature in top_features:
            # Get mean values for each stage
            stage_means = (
                df[df["testmode"] == mode].groupby("teststage")[feature].mean()
            )

            # Calculate rate of change between consecutive stages
            rates = stage_means.pct_change().dropna()

            # Store in dataframe
            temp_df = pd.DataFrame(
                {
                    "testmode": mode,
                    "feature": feature,
                    "teststage": rates.index,
                    "rate_of_change": rates.values,
                }
            )

            rate_of_change_df = pd.concat([rate_of_change_df, temp_df])

    # Plot the rate of change for top features
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(top_features):
        plt.subplot(len(top_features), 1, i + 1)

        feature_data = rate_of_change_df[rate_of_change_df["feature"] == feature]

        for mode in sorted(df["testmode"].unique()):
            mode_data = feature_data[feature_data["testmode"] == mode]
            plt.plot(
                mode_data["teststage"],
                mode_data["rate_of_change"],
                marker="o",
                label=f"Mode {mode}",
            )

        plt.title(f"Rate of Change for {feature}")
        plt.ylabel("Rate of Change")
        plt.grid(True, linestyle="--", alpha=0.7)

        if i == 0:
            plt.legend()

    plt.xlabel("Test Stage")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rate_of_change.png"))
    plt.close()

    print("Time series analysis complete!")


if __name__ == "__main__":
    perform_time_series_analysis()
