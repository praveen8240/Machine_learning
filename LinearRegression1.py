from sklearn.linear_model import LinearRegression

# Reshape x to be a 2D array as required by scikit-learn
x = x.reshape(-1, 1)

# Initialize the model and fit the data
model = LinearRegression()
model.fit(x, y)

# Get the coefficients
print("Intercept (β0):", model.intercept_)
print("Slope (β1):", model.coef_[0])

# Make predictions
y_pred = model.predict(x)
print("Predicted y values:", y_pred)
