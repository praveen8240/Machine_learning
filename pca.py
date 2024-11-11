import numpy as np
from sklearn.preprocessing import StandardScaler

# Example dataset
X = np.array([[2.5, 2.4],
              [0.5, 0.7],
              [2.2, 2.9],
              [1.9, 2.2],
              [3.1, 3.0],
              [2.3, 2.7],
              [2.0, 1.6],
              [1.0, 1.1],
              [1.5, 1.6],
              [1.1, 0.9]])

# Step 1: Standardize the Data
X_standardized = StandardScaler().fit_transform(X)

# Step 2: Compute Covariance Matrix
cov_matrix = np.cov(X_standardized, rowvar=False)

# Step 3: Compute Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 4: Sort Eigenvalues and Eigenvectors
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 5: Select Top k Eigenvectors (for 1 principal component)
k = 1
principal_components = eigenvectors[:, :k]

# Step 6: Transform Data
X_reduced = np.dot(X_standardized, principal_components)

print("Reduced Data:\n", X_reduced)
