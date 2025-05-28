import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif


def train_logistic_regression(
    X_train, X_test, y_train, y_test, all_features, output_dir
):
    """
    Train and evaluate a Logistic Regression model for test mode prediction

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
        Dictionary containing the model, accuracy, predictions, and coefficients
    """
    print("\n=== Training Logistic Regression ===")

    # Create a pipeline with feature selection, scaling, and classifier
    lr_pipeline = Pipeline(
        [
            ("feature_selection", SelectKBest(f_classif, k=10)),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
        ]
    )

    # Define hyperparameters to tune for Logistic Regression
    lr_param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["liblinear", "saga"],
        "feature_selection__k": [5, 10, 15],
    }

    # Perform grid search with cross-validation
    lr_grid_search = GridSearchCV(
        lr_pipeline, lr_param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
    )
    lr_grid_search.fit(X_train, y_train)

    # Print best parameters
    print("\nBest parameters for Logistic Regression:", lr_grid_search.best_params_)

    # Get the best model
    lr_best_model = lr_grid_search.best_estimator_

    # Evaluate on the test set
    lr_y_pred = lr_best_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_y_pred)
    print(f"\nLogistic Regression test accuracy: {lr_accuracy:.4f}")

    # Print classification report
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, lr_y_pred))

    # Create confusion matrix
    plt.figure(figsize=(8, 6))
    lr_cm = confusion_matrix(y_test, lr_y_pred)
    sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Oranges")
    plt.title("Logistic Regression Confusion Matrix")
    plt.ylabel("True Test Mode")
    plt.xlabel("Predicted Test Mode")
    plt.savefig(os.path.join(output_dir, "lr_confusion_matrix.png"))
    plt.close()

    # Get coefficients for Logistic Regression
    lr_feature_selector = lr_best_model.named_steps["feature_selection"]
    lr_classifier = lr_best_model.named_steps["classifier"]
    lr_selected_indices = lr_feature_selector.get_support(indices=True)
    lr_selected_features = [all_features[i] for i in lr_selected_indices]

    # For multi-class, there are coefficients for each class
    lr_coeffs = lr_classifier.coef_

    # Create a DataFrame to store coefficients for each class
    lr_coeffs_df = pd.DataFrame()
    for i, mode in enumerate(lr_classifier.classes_):
        class_coeffs = pd.DataFrame(
            {"Feature": lr_selected_features, f"Coefficient_Mode_{mode}": lr_coeffs[i]}
        )
        lr_coeffs_df = (
            pd.merge(lr_coeffs_df, class_coeffs, on="Feature", how="outer")
            if not lr_coeffs_df.empty
            else class_coeffs
        )

    # Calculate absolute values for overall importance
    lr_coeffs_df["Absolute_Importance"] = lr_coeffs_df.iloc[:, 1:].abs().mean(axis=1)
    lr_coeffs_df = lr_coeffs_df.sort_values("Absolute_Importance", ascending=False)

    # Plot coefficients for Logistic Regression
    plt.figure(figsize=(12, 8))
    sns.barplot(x="Absolute_Importance", y="Feature", data=lr_coeffs_df)
    plt.title("Logistic Regression Feature Importance (Absolute Coefficient Values)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lr_feature_importance.png"))
    plt.close()

    # Plot coefficients by class
    plt.figure(figsize=(14, 10))
    coef_df_long = pd.melt(
        lr_coeffs_df,
        id_vars=["Feature"],
        value_vars=[col for col in lr_coeffs_df.columns if "Coefficient_Mode_" in col],
        var_name="Test Mode",
        value_name="Coefficient",
    )

    # Clean up the mode names
    coef_df_long["Test Mode"] = coef_df_long["Test Mode"].str.replace(
        "Coefficient_Mode_", "Mode "
    )

    # Plot
    sns.barplot(x="Coefficient", y="Feature", hue="Test Mode", data=coef_df_long)
    plt.title("Logistic Regression Coefficients by Test Mode")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lr_coefficients_by_mode.png"))
    plt.close()

    # Return results
    return {
        "model": lr_best_model,
        "accuracy": lr_accuracy,
        "predictions": lr_y_pred,
        "coefficients": lr_coeffs_df,
    }
