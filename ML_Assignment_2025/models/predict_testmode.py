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

# Set font properties for better display
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "output", "testmode_prediction")
os.makedirs(output_dir, exist_ok=True)


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


def train_and_evaluate_model(X_train, X_test, y_train, y_test, all_features):
    """Train and evaluate a model for test mode prediction"""
    # Create a pipeline with feature selection, scaling, and classifier
    pipeline = Pipeline(
        [
            ("feature_selection", SelectKBest(f_classif, k=10)),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Define hyperparameters to tune
    param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 10, 20],
        "feature_selection__k": [5, 10, 15],
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    # Print best parameters
    print("\nBest parameters:", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Evaluate on the test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest set accuracy: {accuracy:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Test Mode")
    plt.xlabel("Predicted Test Mode")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Get feature importances
    # Extract feature selector and trained classifier from pipeline
    feature_selector = best_model.named_steps["feature_selection"]
    classifier = best_model.named_steps["classifier"]

    # Get selected feature indices
    selected_indices = feature_selector.get_support(indices=True)
    selected_features = [all_features[i] for i in selected_indices]

    # Get importances of selected features
    importances = classifier.feature_importances_
    feature_importances = pd.DataFrame(
        {"Feature": selected_features, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=feature_importances)
    plt.title("Feature Importances for Test Mode Prediction")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importances.png"))
    plt.close()

    return best_model, feature_importances


def main():
    """Main function to train and evaluate the test mode prediction model"""
    print("Starting test mode prediction model training...")

    # Load and prepare data
    X_train, X_test, y_train, y_test, all_features = load_and_prepare_data()

    # Train and evaluate model
    best_model, feature_importances = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, all_features
    )

    print("\nTop 5 most important features:")
    print(feature_importances.head(5))

    print("\nModel training and evaluation complete.")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
