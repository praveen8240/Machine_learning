from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load a dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Labels

# Use only two classes for binary classification
X = X[y != 2]
y = y[y != 2]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the SVM model with a linear kernel
svm = SVC(kernel='linear', C=1.0)

# Train the model
svm.fit(X_train, y_train)

# Predict on test data
y_pred = svm.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
