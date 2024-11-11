import numpy as np

# Example matrix
A = np.array([[3, 1, 1], 
              [-1, 3, 1]])

# Compute SVD
U, Sigma, Vt = np.linalg.svd(A)

print("U matrix:\n", U)
print("\nSigma values (as a vector):\n", Sigma)
print("\nV^T matrix:\n", Vt)



# Convert Sigma to a diagonal matrix
Sigma_full = np.zeros((A.shape[0], A.shape[1]))
np.fill_diagonal(Sigma_full, Sigma)

# Reconstruct the matrix A
A_reconstructed = np.dot(U, np.dot(Sigma_full, Vt))

print("\nReconstructed Matrix A:\n", A_reconstructed)



# Reduce dimensions by keeping only the largest singular value
k = 1  # Number of singular values to keep
U_reduced = U[:, :k]
Sigma_reduced = np.diag(Sigma[:k])
Vt_reduced = Vt[:k, :]

# Reconstruct reduced matrix A
A_reduced = np.dot(U_reduced, np.dot(Sigma_reduced, Vt_reduced))

print("\nReduced Matrix A (k=1):\n", A_reduced)
