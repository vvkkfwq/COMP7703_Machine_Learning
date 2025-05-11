import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('./w3regr.csv', header=None, names=['X', 'Y'])

# Shuffle the dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the dataset into features and labels
X = data[['X']]
y = data['Y']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a decision tree regressor
tree_regressor = DecisionTreeRegressor(random_state=42)

# Train the regressor
tree_regressor.fit(X_train, y_train)

# Predict on training and testing sets
y_train_pred = tree_regressor.predict(X_train)
y_test_pred = tree_regressor.predict(X_test)

# Calculate sum of squared error for training and testing sets
train_loss = mean_squared_error(y_train, y_train_pred) * len(y_train)
test_loss = mean_squared_error(y_test, y_test_pred) * len(y_test)

print(f'Training Loss (Sum of Squared Error): {train_loss:.4f}')
print(f'Test Loss (Sum of Squared Error): {test_loss:.4f}')

# Plot the training and test data together with the predicted function
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(X_train, y_train, color='blue', label='Training Data')

# Plot test data
plt.scatter(X_test, y_test, color='green', label='Testing Data')

# Plot predicted function
X_plot = np.linspace(X.min(), X.max(), 500).reshape(-1, 1)
y_plot = tree_regressor.predict(X_plot)
plt.plot(X_plot, y_plot, color='red', label='Predicted Function')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Decision Tree Regression')
plt.legend()
plt.grid(True)
plt.show()