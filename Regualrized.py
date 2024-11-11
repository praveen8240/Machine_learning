import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Initialize Ridge with a chosen regularization parameter
ridge = Ridge(alpha=1.0)  # alpha is the λ parameter
ridge.fit(X_train, y_train)

# Predictions and performance
y_pred_ridge = ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, y_pred_ridge)
print("Ridge MSE:", ridge_mse)


from sklearn.linear_model import Lasso

# Initialize Lasso with a chosen regularization parameter
lasso = Lasso(alpha=0.1)  # alpha is the λ parameter
lasso.fit(X_train, y_train)

# Predictions and performance
y_pred_lasso = lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, y_pred_lasso)
print("Lasso MSE:", lasso_mse)

# Check coefficients
print("Lasso Coefficients:", lasso.coef_)


from sklearn.linear_model import ElasticNet

# Initialize Elastic Net with chosen regularization parameters
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)  # alpha is the λ parameter, l1_ratio controls L1/L2 mix
elastic_net.fit(X_train, y_train)

# Predictions and performance
y_pred_elastic = elastic_net.predict(X_test)
elastic_mse = mean_squared_error(y_test, y_pred_elastic)
print("Elastic Net MSE:", elastic_mse)
