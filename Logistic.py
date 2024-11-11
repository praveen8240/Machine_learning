import numpy as np

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Binary Cross-Entropy Loss
def compute_cost(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

# Gradient Descent
def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0

    for _ in range(epochs):
        # Linear model
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)

        # Calculate gradients
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Update weights
        weights -= learning_rate * dw
        bias -= learning_rate * db

    return weights, bias

# Example data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# Train model
weights, bias = gradient_descent(X, y)
print("Weights:", weights)
print("Bias:", bias)

# Make predictions
def predict(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias) >= 0.5

print("Predictions:", predict(X, weights, bias))
