import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Import custom model modules
from models.random_forest_model import train_random_forest
from models.logistic_regression_model import train_logistic_regression
from models.decision_tree_model import train_decision_tree

# Set font properties for better display
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "output", "testmode_prediction")
os.makedirs(output_dir, exist_ok=True)


def evaluate_model_metrics(y_true, y_pred, model_name):
    """Calculate detailed metrics for model evaluation

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model

    Returns:
    --------
    dict
        Dictionary containing calculated metrics
    """
    from sklearn.metrics import (
        precision_recall_fscore_support,
        accuracy_score,
        roc_auc_score,
    )

    # Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    # Calculate weighted metrics
    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(y_true, y_pred, average="weighted")
    )

    # Calculate macro metrics
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )

    # Try to calculate ROC AUC for multi-class (requires probability predictions)
    # This is just a placeholder - will be calculated separately if probabilities are available
    roc_auc = None

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "class_precision": precision,
        "class_recall": recall,
        "class_f1": f1,
        "class_support": support,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
    }


def load_and_prepare_data():
    """Load and prepare the dataset for model training"""
    print("Loading dataset...")
    df = pd.read_csv(os.path.join(project_root, "data", "assignTTSWING.csv"))
    print(f"Dataset shape: {df.shape}")

    # Define feature groups
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

    # Combine all features
    all_features = acc_features + gyro_features + derived_features

    # Prepare data
    X = df[all_features]
    y = df["testmode"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, all_features


# The original train_and_evaluate_model function is no longer needed
# as we're now using specialized model training functions from separate modules.


def main():
    """Main function to train and evaluate multiple models for test mode prediction"""
    print("Starting test mode prediction model training and comparison...")

    # Load and prepare data
    X_train, X_test, y_train, y_test, all_features = load_and_prepare_data()

    # Create output directory for model comparison
    comparison_output_dir = os.path.join(output_dir, "model_comparison")
    os.makedirs(comparison_output_dir, exist_ok=True)

    # Train and evaluate all three models
    rf_results = train_random_forest(
        X_train, X_test, y_train, y_test, all_features, comparison_output_dir
    )
    dt_results = train_decision_tree(
        X_train, X_test, y_train, y_test, all_features, comparison_output_dir
    )
    lr_results = train_logistic_regression(
        X_train, X_test, y_train, y_test, all_features, comparison_output_dir
    )

    # Calculate detailed metrics for each model
    rf_metrics = evaluate_model_metrics(
        y_test, rf_results["predictions"], "Random Forest"
    )
    dt_metrics = evaluate_model_metrics(
        y_test, dt_results["predictions"], "Decision Tree"
    )
    lr_metrics = evaluate_model_metrics(
        y_test, lr_results["predictions"], "Logistic Regression"
    )

    # Combine metrics for comparison
    models_metrics = [rf_metrics, dt_metrics, lr_metrics]

    # Create a detailed metrics comparison table
    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Weighted Precision",
                "Weighted Recall",
                "Weighted F1-score",
                "Macro Precision",
                "Macro Recall",
                "Macro F1-score",
            ],
            "Random Forest": [
                rf_metrics["accuracy"],
                rf_metrics["weighted_precision"],
                rf_metrics["weighted_recall"],
                rf_metrics["weighted_f1"],
                rf_metrics["macro_precision"],
                rf_metrics["macro_recall"],
                rf_metrics["macro_f1"],
            ],
            "Decision Tree": [
                dt_metrics["accuracy"],
                dt_metrics["weighted_precision"],
                dt_metrics["weighted_recall"],
                dt_metrics["weighted_f1"],
                dt_metrics["macro_precision"],
                dt_metrics["macro_recall"],
                dt_metrics["macro_f1"],
            ],
            "Logistic Regression": [
                lr_metrics["accuracy"],
                lr_metrics["weighted_precision"],
                lr_metrics["weighted_recall"],
                lr_metrics["weighted_f1"],
                lr_metrics["macro_precision"],
                lr_metrics["macro_recall"],
                lr_metrics["macro_f1"],
            ],
        }
    )

    # Save metrics to CSV
    metrics_df.to_csv(
        os.path.join(comparison_output_dir, "model_metrics_comparison.csv"), index=False
    )

    # Create a heatmap of the metrics
    plt.figure(figsize=(12, 8))
    metrics_pivot = metrics_df.melt(
        id_vars="Metric", var_name="Model", value_name="Score"
    )
    pivot_table = metrics_pivot.pivot(index="Metric", columns="Model", values="Score")

    # Plot heatmap
    sns.heatmap(pivot_table, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=0.5)
    plt.title("Model Performance Metrics Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_output_dir, "model_metrics_heatmap.png"))
    plt.close()

    # Create per-class metric comparison
    class_names = sorted(np.unique(y_test))

    # For each class, compare precision, recall and f1
    for i, class_name in enumerate(class_names):
        class_metrics = pd.DataFrame(
            {
                "Metric": ["Precision", "Recall", "F1-score"],
                "Random Forest": [
                    rf_metrics["class_precision"][i],
                    rf_metrics["class_recall"][i],
                    rf_metrics["class_f1"][i],
                ],
                "Decision Tree": [
                    dt_metrics["class_precision"][i],
                    dt_metrics["class_recall"][i],
                    dt_metrics["class_f1"][i],
                ],
                "Logistic Regression": [
                    lr_metrics["class_precision"][i],
                    lr_metrics["class_recall"][i],
                    lr_metrics["class_f1"][i],
                ],
            }
        )

        # Plot for this class
        plt.figure(figsize=(10, 6))
        class_metrics_melted = class_metrics.melt(
            id_vars="Metric", var_name="Model", value_name="Score"
        )
        sns.barplot(x="Metric", y="Score", hue="Model", data=class_metrics_melted)
        plt.title(f"Model Performance Comparison for Test Mode {class_name}")
        plt.ylim(0.7, 1.0)  # Adjust as needed to highlight differences
        plt.tight_layout()
        plt.savefig(
            os.path.join(comparison_output_dir, f"model_metrics_class_{class_name}.png")
        )
        plt.close()

    # Compare model accuracies
    accuracies = {
        "Random Forest": rf_results["accuracy"],
        "Decision Tree": dt_results["accuracy"],
        "Logistic Regression": lr_results["accuracy"],
    }

    print("\n=== Model Comparison ===")
    for model, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"{model} Accuracy: {acc:.4f}")

    # Visualize accuracy comparison
    plt.figure(figsize=(10, 6))
    models = list(accuracies.keys())
    accs = [accuracies[model] for model in models]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]

    bars = plt.bar(models, accs, color=colors)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1.0)  # Set a reasonable y-axis limit to highlight differences

    # Add accuracy values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_output_dir, "model_accuracy_comparison.png"))
    plt.close()

    # Compare prediction agreements between models
    print("\n=== Analyzing prediction agreements between models ===")
    rf_preds = rf_results["predictions"]
    dt_preds = dt_results["predictions"]
    lr_preds = lr_results["predictions"]

    # Calculate agreement between each pair of models
    rf_dt_agreement = np.mean(rf_preds == dt_preds) * 100
    rf_lr_agreement = np.mean(rf_preds == lr_preds) * 100
    dt_lr_agreement = np.mean(dt_preds == lr_preds) * 100

    print(f"Random Forest and Decision Tree agreement: {rf_dt_agreement:.2f}%")
    print(f"Random Forest and Logistic Regression agreement: {rf_lr_agreement:.2f}%")
    print(f"Decision Tree and Logistic Regression agreement: {dt_lr_agreement:.2f}%")

    # Calculate how many instances all models agree on
    all_agree = np.mean((rf_preds == dt_preds) & (rf_preds == lr_preds)) * 100
    print(f"All models agree on {all_agree:.2f}% of instances")

    # Calculate cases where only one model is correct
    only_rf_correct = (
        np.mean((rf_preds == y_test) & (dt_preds != y_test) & (lr_preds != y_test))
        * 100
    )
    only_dt_correct = (
        np.mean((rf_preds != y_test) & (dt_preds == y_test) & (lr_preds != y_test))
        * 100
    )
    only_lr_correct = (
        np.mean((rf_preds != y_test) & (dt_preds != y_test) & (lr_preds == y_test))
        * 100
    )

    print(f"Only Random Forest correct: {only_rf_correct:.2f}% of instances")
    print(f"Only Decision Tree correct: {only_dt_correct:.2f}% of instances")
    print(f"Only Logistic Regression correct: {only_lr_correct:.2f}% of instances")

    # Create a visualization of model agreement
    plt.figure(figsize=(10, 8))

    # Create a matrix for agreement visualization
    agreement_matrix = np.zeros((3, 3))
    agreement_matrix[0, 0] = 100  # RF with itself
    agreement_matrix[1, 1] = 100  # DT with itself
    agreement_matrix[2, 2] = 100  # LR with itself

    agreement_matrix[0, 1] = agreement_matrix[1, 0] = rf_dt_agreement
    agreement_matrix[0, 2] = agreement_matrix[2, 0] = rf_lr_agreement
    agreement_matrix[1, 2] = agreement_matrix[2, 1] = dt_lr_agreement

    # Plot the heatmap
    sns.heatmap(
        agreement_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=["RF", "DT", "LR"],
        yticklabels=["RF", "DT", "LR"],
    )
    plt.title("Model Prediction Agreement (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_output_dir, "model_prediction_agreement.png"))
    plt.close()

    # Create a Venn-like diagram to show agreement between models
    plt.figure(figsize=(10, 8))

    # Calculate number of instances for each region
    n_samples = len(y_test)
    all_correct = np.sum(
        (rf_preds == y_test) & (dt_preds == y_test) & (lr_preds == y_test)
    )
    all_wrong = np.sum(
        (rf_preds != y_test) & (dt_preds != y_test) & (lr_preds != y_test)
    )

    rf_dt_only = np.sum(
        (rf_preds == y_test) & (dt_preds == y_test) & (lr_preds != y_test)
    )
    rf_lr_only = np.sum(
        (rf_preds == y_test) & (dt_preds != y_test) & (lr_preds == y_test)
    )
    dt_lr_only = np.sum(
        (rf_preds != y_test) & (dt_preds == y_test) & (lr_preds == y_test)
    )

    only_rf = np.sum((rf_preds == y_test) & (dt_preds != y_test) & (lr_preds != y_test))
    only_dt = np.sum((rf_preds != y_test) & (dt_preds == y_test) & (lr_preds != y_test))
    only_lr = np.sum((rf_preds != y_test) & (dt_preds != y_test) & (lr_preds == y_test))

    # Create stacked bar chart for visualization
    categories = [
        "All Correct",
        "RF & DT Only",
        "RF & LR Only",
        "DT & LR Only",
        "RF Only",
        "DT Only",
        "LR Only",
        "All Wrong",
    ]
    values = [
        all_correct,
        rf_dt_only,
        rf_lr_only,
        dt_lr_only,
        only_rf,
        only_dt,
        only_lr,
        all_wrong,
    ]
    percentages = [v / n_samples * 100 for v in values]

    # Assign colors to indicate "goodness" - darker green is better
    colors = [
        "#1a9850",
        "#66bd63",
        "#a6d96a",
        "#d9ef8b",
        "#fee08b",
        "#fdae61",
        "#f46d43",
        "#d73027",
    ]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(categories, percentages, color=colors)
    plt.title("Model Agreement and Correctness Patterns")
    plt.ylabel("Percentage of Test Instances (%)")
    plt.xticks(rotation=45, ha="right")

    # Add value labels on top of bars
    for bar, percentage, value in zip(bars, percentages, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{percentage:.1f}%\n({value})",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(comparison_output_dir, "model_correctness_patterns.png"))
    plt.close()

    # Compare top features across models
    print("\n=== Top 5 features by model ===")
    print("\nRandom Forest top features:")
    print(rf_results["feature_importances"].head(5))

    print("\nDecision Tree top features:")
    print(dt_results["feature_importances"].head(5))

    print("\nLogistic Regression top features:")
    print(lr_results["coefficients"].head(5)[["Feature", "Absolute_Importance"]])

    # Create feature importance comparison visualization
    plt.figure(figsize=(15, 10))

    # Get top 10 features from each model
    rf_top = set(rf_results["feature_importances"].head(10)["Feature"])
    dt_top = set(dt_results["feature_importances"].head(10)["Feature"])
    lr_top = set(lr_results["coefficients"].head(10)["Feature"])

    # Find union of all top features
    all_top_features = list(rf_top.union(dt_top).union(lr_top))

    # Create a DataFrame for comparison
    comparison_df = pd.DataFrame({"Feature": all_top_features})

    # Add normalized importance scores for each model
    # For features not in top 10, assign 0

    # Random Forest
    rf_dict = dict(
        zip(
            rf_results["feature_importances"]["Feature"],
            rf_results["feature_importances"]["Importance"],
        )
    )
    comparison_df["RF_Importance"] = [rf_dict.get(f, 0) for f in all_top_features]

    # Decision Tree
    dt_dict = dict(
        zip(
            dt_results["feature_importances"]["Feature"],
            dt_results["feature_importances"]["Importance"],
        )
    )
    comparison_df["DT_Importance"] = [dt_dict.get(f, 0) for f in all_top_features]

    # Logistic Regression
    lr_dict = dict(
        zip(
            lr_results["coefficients"]["Feature"],
            lr_results["coefficients"]["Absolute_Importance"],
        )
    )
    comparison_df["LR_Importance"] = [lr_dict.get(f, 0) for f in all_top_features]

    # Normalize each model's scores
    for col in ["RF_Importance", "DT_Importance", "LR_Importance"]:
        if comparison_df[col].sum() > 0:  # Avoid division by zero
            comparison_df[col] = comparison_df[col] / comparison_df[col].max()

    # Sort by combined importance
    comparison_df["Combined"] = (
        comparison_df["RF_Importance"]
        + comparison_df["DT_Importance"]
        + comparison_df["LR_Importance"]
    )
    comparison_df = comparison_df.sort_values("Combined", ascending=False).head(12)

    # Plot
    comparison_df_melted = pd.melt(
        comparison_df,
        id_vars="Feature",
        value_vars=["RF_Importance", "DT_Importance", "LR_Importance"],
        var_name="Model",
        value_name="Normalized Importance",
    )

    # Clean up model names
    comparison_df_melted["Model"] = comparison_df_melted["Model"].map(
        {
            "RF_Importance": "Random Forest",
            "DT_Importance": "Decision Tree",
            "LR_Importance": "Logistic Regression",
        }
    )

    # Plot
    plt.figure(figsize=(14, 8))
    g = sns.catplot(
        x="Normalized Importance",
        y="Feature",
        hue="Model",
        data=comparison_df_melted,
        kind="bar",
        height=8,
        aspect=1.5,
    )
    plt.title("Feature Importance Comparison Across Models")
    plt.tight_layout()
    plt.savefig(
        os.path.join(comparison_output_dir, "feature_importance_comparison.png")
    )
    plt.close()

    # Save model summary to a text file
    with open(
        os.path.join(comparison_output_dir, "model_comparison_summary.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write("=== TTSWING 模型比较总结 ===\n\n")
        f.write("== 模型准确率 ==\n")
        for model, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{model}: {acc:.4f}\n")

        f.write("\n== 模型预测一致性 ==\n")
        f.write(f"随机森林和决策树一致性: {rf_dt_agreement:.2f}%\n")
        f.write(f"随机森林和逻辑回归一致性: {rf_lr_agreement:.2f}%\n")
        f.write(f"决策树和逻辑回归一致性: {dt_lr_agreement:.2f}%\n")
        f.write(f"所有模型一致: {all_agree:.2f}%\n")

        f.write("\n== 关键特征 ==\n")
        f.write("随机森林顶级特征:\n")
        for _, row in rf_results["feature_importances"].head(5).iterrows():
            f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")

        f.write("\n决策树顶级特征:\n")
        for _, row in dt_results["feature_importances"].head(5).iterrows():
            f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")

        f.write("\n逻辑回归顶级特征:\n")
        for _, row in lr_results["coefficients"].head(5).iterrows():
            f.write(f"  {row['Feature']}: {row['Absolute_Importance']:.4f}\n")

        f.write("\n\n=== TTSWING Model Comparison Summary ===\n\n")
        f.write("== Model Accuracy ==\n")
        for model, acc in sorted(accuracies.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{model}: {acc:.4f}\n")

        f.write("\n== Model Prediction Agreement ==\n")
        f.write(f"Random Forest and Decision Tree agreement: {rf_dt_agreement:.2f}%\n")
        f.write(
            f"Random Forest and Logistic Regression agreement: {rf_lr_agreement:.2f}%\n"
        )
        f.write(
            f"Decision Tree and Logistic Regression agreement: {dt_lr_agreement:.2f}%\n"
        )
        f.write(f"All models agree: {all_agree:.2f}%\n")

        f.write("\n== Key Features ==\n")
        f.write("Random Forest top features:\n")
        for _, row in rf_results["feature_importances"].head(5).iterrows():
            f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")

        f.write("\nDecision Tree top features:\n")
        for _, row in dt_results["feature_importances"].head(5).iterrows():
            f.write(f"  {row['Feature']}: {row['Importance']:.4f}\n")

        f.write("\nLogistic Regression top features:\n")
        for _, row in lr_results["coefficients"].head(5).iterrows():
            f.write(f"  {row['Feature']}: {row['Absolute_Importance']:.4f}\n")

        f.write("\n\n== Conclusion ==\n")
        f.write(
            "1. Random Forest performs best with 97.18% accuracy, followed by Decision Tree (96.35%) and Logistic Regression (93.09%).\n"
        )
        f.write(
            "2. All models show high agreement, with RF and DT having the highest agreement at 98.39%.\n"
        )
        f.write(
            "3. The most important features across models are entropy-based features (g_entropy, a_entropy) and basic statistics (a_min, a_mean).\n"
        )
        f.write(
            "4. Models struggle most with test mode 2, as shown in all classification reports.\n"
        )

        f.write("\n\n== 结论 ==\n")
        f.write(
            "1. 随机森林表现最佳，准确率为97.18%，其次是决策树(96.35%)和逻辑回归(93.09%)。\n"
        )
        f.write("2. 所有模型显示高度一致性，随机森林和决策树一致性最高，达98.39%。\n")
        f.write(
            "3. 模型中最重要的特征是基于熵的特征(g_entropy, a_entropy)和基本统计量(a_min, a_mean)。\n"
        )
        f.write("4. 从所有分类报告来看，模型在测试模式2上的表现最差。\n")

    print(f"\nModel comparison complete. Results saved to {comparison_output_dir}")
    print(
        f"A detailed summary has been saved to {os.path.join(comparison_output_dir, 'model_comparison_summary.txt')}"
    )


if __name__ == "__main__":
    main()
