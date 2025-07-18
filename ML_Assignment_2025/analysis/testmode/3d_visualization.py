import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Set font properties for better display
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Create output directory if it doesn't exist
output_dir = os.path.join(
    project_root, "output", "testmode_analysis", "3d_visualization"
)
os.makedirs(output_dir, exist_ok=True)


def create_3d_visualizations():
    """Create 3D visualizations to show relationships between top features and test modes"""
    print("Loading dataset...")
    df = pd.read_csv(os.path.join(project_root, "data", "assignTTSWING.csv"))
    print(f"Dataset shape: {df.shape}")

    # Based on previous analysis, the top features were identified
    # Here we'll use the top 3 features for 3D visualization
    top_features = ["a_min", "a_max", "g_max"]  # Adjust based on your actual findings

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Define colors for different test modes
    colors = ["blue", "red", "green"]

    # Plot each test mode with a different color
    for mode, color in zip(sorted(df["testmode"].unique()), colors):
        mode_data = df[df["testmode"] == mode]
        ax.scatter(
            mode_data[top_features[0]],
            mode_data[top_features[1]],
            mode_data[top_features[2]],
            c=color,
            label=f"Mode {mode}",
            alpha=0.6,
            s=50,
        )

    ax.set_xlabel(top_features[0])
    ax.set_ylabel(top_features[1])
    ax.set_zlabel(top_features[2])
    ax.set_title("3D Visualization of Top 3 Features for Test Mode Classification")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "3d_feature_visualization.png"))
    plt.close()

    # Create pairwise scatterplots with density contours
    print("Creating pairwise scatter plots with density contours...")

    # Add a categorical color column for easier seaborn plotting
    df["testmode_cat"] = df["testmode"].astype("category")

    # Create pair plots
    plt.figure(figsize=(15, 15))
    pair_plot = sns.pairplot(
        df,
        vars=top_features,
        hue="testmode_cat",
        palette="viridis",
        diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 50, "edgecolor": "k"},
        height=3,
    )
    pair_plot.fig.suptitle(
        "Pairwise Relationships Between Top Features", y=1.02, fontsize=16
    )
    plt.savefig(os.path.join(output_dir, "pairwise_feature_relationships.png"))
    plt.close()

    # Create a 3D animated visualization (save multiple angles)
    print("Creating multi-angle 3D visualization...")
    for angle in range(0, 360, 30):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        for mode, color in zip(sorted(df["testmode"].unique()), colors):
            mode_data = df[df["testmode"] == mode]
            ax.scatter(
                mode_data[top_features[0]],
                mode_data[top_features[1]],
                mode_data[top_features[2]],
                c=color,
                label=f"Mode {mode}",
                alpha=0.6,
                s=50,
            )

        ax.set_xlabel(top_features[0])
        ax.set_ylabel(top_features[1])
        ax.set_zlabel(top_features[2])
        ax.set_title("3D Visualization of Top 3 Features (Multiple Angles)")

        # Set the viewing angle
        ax.view_init(30, angle)

        plt.legend()
        plt.savefig(os.path.join(output_dir, f"3d_feature_angle_{angle}.png"))
        plt.close()

    print("3D visualizations complete!")


if __name__ == "__main__":
    create_3d_visualizations()
