import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif


def train_random_forest(X_train, X_test, y_train, y_test, all_features, output_dir):
    """
    Train and evaluate a Random Forest model for test mode prediction

    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Testing features
    y_train : Series
        Training labels
    y_test : Series
        Testing labels
    all_features : list
        List of all feature names
    output_dir : str
        Directory to save outputs

    Returns:
    --------
    dict
        Dictionary containing the model, accuracy, predictions, and feature importances
    """
    print("\n=== Training Random Forest Classifier ===")

    # Create a pipeline with feature selection, scaling, and classifier
    rf_pipeline = Pipeline(
        [
            ("feature_selection", SelectKBest(f_classif, k=10)),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Define hyperparameters to tune for Random Forest
    rf_param_grid = {
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [None, 10, 20],
        "feature_selection__k": [5, 10, 15],
    }

    # Perform grid search with cross-validation
    rf_grid_search = GridSearchCV(
        rf_pipeline, rf_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    rf_grid_search.fit(X_train, y_train)

    # Print best parameters
    print("\nBest parameters for Random Forest:", rf_grid_search.best_params_)

    # Get the best model
    rf_best_model = rf_grid_search.best_estimator_

    # Evaluate on the test set
    rf_y_pred = rf_best_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_y_pred)
    print(f"\nRandom Forest test accuracy: {rf_accuracy:.4f}")

    # Print classification report
    print("\nRandom Forest Classification Report:")
    print(classification_report(y_test, rf_y_pred))

    # Create confusion matrix
    plt.figure(figsize=(8, 6))
    rf_cm = confusion_matrix(y_test, rf_y_pred)
    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Random Forest Confusion Matrix")
    plt.ylabel("True Test Mode")
    plt.xlabel("Predicted Test Mode")
    plt.savefig(os.path.join(output_dir, "rf_confusion_matrix.png"))
    plt.close()

    # Get feature importances for Random Forest
    rf_feature_selector = rf_best_model.named_steps["feature_selection"]
    rf_classifier = rf_best_model.named_steps["classifier"]
    rf_selected_indices = rf_feature_selector.get_support(indices=True)
    rf_selected_features = [all_features[i] for i in rf_selected_indices]
    rf_importances = rf_classifier.feature_importances_
    rf_feature_importances = pd.DataFrame(
        {"Feature": rf_selected_features, "Importance": rf_importances}
    ).sort_values("Importance", ascending=False)

    # Plot feature importances for Random Forest
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=rf_feature_importances)
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rf_feature_importances.png"))
    plt.close()

    # Return results
    return {
        "model": rf_best_model,
        "accuracy": rf_accuracy,
        "predictions": rf_y_pred,
        "feature_importances": rf_feature_importances,
    }
