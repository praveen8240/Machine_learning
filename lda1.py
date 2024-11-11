from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Initialize LDA and transform the data
lda = LDA(n_components=1)  # For 1-dimensional projection
X_lda = lda.fit_transform(X, y)

print("Transformed data (Scikit-Learn):\n", X_lda)
