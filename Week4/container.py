import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Read the CSV file
df = pd.read_csv("w3classif.csv")
df.columns = ["feature1", "feature2", "target"]

# Separate features and target
X = df[["feature1", "feature2"]]
y = df["target"]

# Fit a logistic regression model to the data
model = LogisticRegression(solver="lbfgs")
model.fit(X, y)

# Print the model parameter values
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Create a DataFrame for the test point with valid feature names
x_test_df = pd.DataFrame([[1.1, 1.1]], columns=["feature1", "feature2"])

# Predict the probability for class 1 for test point x' = (1.1, 1.1)
p = model.predict_proba(x_test_df)[0, 1]
print("Predicted probability p(y'=1 | x'=(1.1,1.1)):", p)

# Plot the data along with the decision regions
# Create a mesh grid covering the feature space
x_min, x_max = X["feature1"].min() - 1, X["feature1"].max() + 1
y_min, y_max = X["feature2"].min() - 1, X["feature2"].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

# Convert grid to DataFrame with valid feature names to avoid warning
grid = np.c_[xx.ravel(), yy.ravel()]
grid_df = pd.DataFrame(grid, columns=["feature1", "feature2"])

# Predict on every point in the grid using grid_df
Z = model.predict(grid_df)
Z = Z.reshape(xx.shape)

# Plot contour for decision regions
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)

# Plot the original data points
plt.scatter(
    X["feature1"],
    X["feature2"],
    c=y,
    edgecolor="k",
    cmap=plt.cm.Paired,
    label="Training Data",
)

# Mark test data point x' = (1.1, 1.1)
plt.scatter(
    x_test_df["feature1"],
    x_test_df["feature2"],
    color="red",
    marker="x",
    s=200,
    label="Test Point (1.1, 1.1)",
)

plt.xlabel("feature1")
plt.ylabel("feature2")
plt.title("Logistic Regression: Decision Regions and Discriminant Function")
plt.legend()
plt.show()
