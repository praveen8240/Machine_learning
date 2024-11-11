import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 1.3, 3.75, 2.25])

# Calculate the mean of x and y
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate the coefficients (slope and intercept)
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)
slope = numerator / denominator
intercept = y_mean - slope * x_mean

print("Slope (β1):", slope)
print("Intercept (β0):", intercept)

# Make predictions
y_pred = intercept + slope * x
print("Predicted y values:", y_pred)
