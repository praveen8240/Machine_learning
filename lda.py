import numpy as np

# Example data
data = np.array([[4.0, 2.0, 0],
                 [2.0, 4.0, 0],
                 [2.0, 3.0, 0],
                 [3.0, 6.0, 0],
                 [4.0, 4.0, 0],
                 [9.0, 10.0, 1],
                 [6.0, 8.0, 1],
                 [9.0, 5.0, 1],
                 [8.0, 7.0, 1],
                 [10.0, 8.0, 1]])

# Separate features and class labels
X = data[:, :2]  # Feature matrix
y = data[:, 2]   # Class labels

# Step 1: Compute the mean vectors for each class
mean_vectors = []
for label in np.unique(y):
    mean_vectors.append(np.mean(X[y == label], axis=0))

# Step 2: Compute the Within-Class Scatter Matrix SW
S_W = np.zeros((2, 2))
for label, mv in zip(np.unique(y), mean_vectors):
    class_scatter = np.cov(X[y == label].T)
    S_W += class_scatter

# Step 3: Compute the Between-Class Scatter Matrix SB
overall_mean = np.mean(X, axis=0)
S_B = np.zeros((2, 2))
for i, mean_vec in enumerate(mean_vectors):
    n = X[y == i, :].shape[0]
    mean_vec = mean_vec.reshape(2, 1)  # make column vector
    overall_mean = overall_mean.reshape(2, 1)  # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

# Step 4: Compute the Eigenvalues and Eigenvectors for SW^-1 * SB
eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Step 5: Select the linear discriminants (eigenvector with highest eigenvalue)
eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

# Create the transformation matrix W with top eigenvectors
W = np.hstack((eigen_pairs[0][1].reshape(2, 1), eigen_pairs[1][1].reshape(2, 1)))

# Step 6: Transform the data
X_lda = X.dot(W)

print("Transformed data:\n", X_lda)
