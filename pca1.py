from sklearn.decomposition import PCA

# Initialize PCA with 1 component
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X_standardized)

print("Reduced Data (Scikit-Learn PCA):\n", X_pca)
