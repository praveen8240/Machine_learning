from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize a weak learner (Decision Tree with max depth 1, a decision stump)
weak_learner = DecisionTreeClassifier(max_depth=1)

# Initialize AdaBoost with 50 weak learners
ada_boost = AdaBoostClassifier(base_estimator=weak_learner, n_estimators=50, random_state=42)

# Train the AdaBoost model
ada_boost.fit(X_train, y_train)

# Make predictions
y_pred = ada_boost.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
