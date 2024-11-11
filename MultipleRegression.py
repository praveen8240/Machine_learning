# Sample data with multiple features
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([1, 2, 1.3, 3.75, 2.25])

# Initialize and fit the model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients
print("Intercept (β0):", model.intercept_)
print("Coefficients (β1, β2, ...):", model.coef_)

# Make predictions
y_pred = model.predict(X)
print("Predicted y values:", y_pred)
