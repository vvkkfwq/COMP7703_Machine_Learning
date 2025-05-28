import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif


def train_decision_tree(X_train, X_test, y_train, y_test, all_features, output_dir):
    """
    Train and evaluate a Decision Tree model for test mode prediction

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
    print("\n=== Training Decision Tree ===")

    # Create a pipeline with feature selection, scaling, and classifier
    dt_pipeline = Pipeline(
        [
            ("feature_selection", SelectKBest(f_classif, k=10)),
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )

    # Define hyperparameters to tune for Decision Tree
    dt_param_grid = {
        "classifier__max_depth": [None, 5, 10, 15, 20],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__criterion": ["gini", "entropy"],
        "feature_selection__k": [5, 10, 15],
    }

    # Perform grid search with cross-validation
    dt_grid_search = GridSearchCV(
        dt_pipeline, dt_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    dt_grid_search.fit(X_train, y_train)

    # Print best parameters
    print("\nBest parameters for Decision Tree:", dt_grid_search.best_params_)

    # Get the best model
    dt_best_model = dt_grid_search.best_estimator_

    # Evaluate on the test set
    dt_y_pred = dt_best_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_y_pred)
    print(f"\nDecision Tree test accuracy: {dt_accuracy:.4f}")

    # Print classification report
    print("\nDecision Tree Classification Report:")
    print(classification_report(y_test, dt_y_pred))

    # Create confusion matrix
    plt.figure(figsize=(8, 6))
    dt_cm = confusion_matrix(y_test, dt_y_pred)
    sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Decision Tree Confusion Matrix")
    plt.ylabel("True Test Mode")
    plt.xlabel("Predicted Test Mode")
    plt.savefig(os.path.join(output_dir, "dt_confusion_matrix.png"))
    plt.close()

    # Get feature importances for Decision Tree
    dt_feature_selector = dt_best_model.named_steps["feature_selection"]
    dt_classifier = dt_best_model.named_steps["classifier"]
    dt_selected_indices = dt_feature_selector.get_support(indices=True)
    dt_selected_features = [all_features[i] for i in dt_selected_indices]
    dt_importances = dt_classifier.feature_importances_
    dt_feature_importances = pd.DataFrame(
        {"Feature": dt_selected_features, "Importance": dt_importances}
    ).sort_values("Importance", ascending=False)

    # Plot feature importances for Decision Tree
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=dt_feature_importances)
    plt.title("Decision Tree Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dt_feature_importances.png"))
    plt.close()

    # Visualize the decision tree (if not too large)
    if dt_classifier.max_depth is not None and dt_classifier.max_depth <= 5:
        plt.figure(figsize=(20, 10))
        plot_tree(
            dt_classifier,
            feature_names=dt_selected_features,
            class_names=[str(i) for i in dt_classifier.classes_],
            filled=True,
            rounded=True,
        )
        plt.title("Decision Tree Visualization")
        plt.savefig(
            os.path.join(output_dir, "decision_tree_visualization.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Return results
    return {
        "model": dt_best_model,
        "accuracy": dt_accuracy,
        "predictions": dt_y_pred,
        "feature_importances": dt_feature_importances,
    }
