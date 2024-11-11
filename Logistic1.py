from sklearn.linear_model import LogisticRegression

# Sample data
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 0, 1, 1, 1])

# Initialize and train the model
model = LogisticRegression()
model.fit(X, y)

# Model parameters
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Predict probabilities
print("Predicted probabilities:", model.predict_proba(X))

# Predict class labels
print("Predicted classes:", model.predict(X))
