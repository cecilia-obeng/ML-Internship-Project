import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# ---------------------------------
# 1. Load Dataset
# ---------------------------------

iris = load_iris()

X = iris.data
y = iris.target

# ---------------------------------
# 2. Split Dataset
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------
# 3. Train Decision Tree Model
# ---------------------------------

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# ---------------------------------
# 4. Make Predictions
# ---------------------------------

predictions = model.predict(X_test)

# ---------------------------------
# 5. Evaluate Model
# ---------------------------------

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ---------------------------------
# 6. Visualize the Tree
# ---------------------------------

plt.figure(figsize=(10,6))
plot_tree(model, filled=True, feature_names=iris.feature_names)
plt.show()