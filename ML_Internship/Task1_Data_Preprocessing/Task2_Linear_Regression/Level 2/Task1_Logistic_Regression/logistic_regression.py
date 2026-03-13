import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------
# 1. Create Dataset
# ---------------------------------

data = {
    "Age": [22, 25, 30, 35, 40, 45, 50, 55, 60, 28, 33, 48],
    "Salary": [15000, 29000, 32000, 35000, 42000, 48000, 52000, 58000, 62000, 27000, 31000, 50000],
    "Purchased": [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1]
}


df = pd.DataFrame(data)

print("Dataset:")
print(df)

# ---------------------------------
# 2. Define Features and Target
# ---------------------------------

X = df[["Age", "Salary"]]
y = df["Purchased"]

# ---------------------------------
# 3. Split Dataset
# ---------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------------------
# 4. Train Logistic Regression Model
# ---------------------------------

model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------------------------
# 5. Make Predictions
# ---------------------------------

predictions = model.predict(X_test)

print("\nPredictions:")
print(predictions)

print("\nActual Values:")
print(y_test.values)

# ---------------------------------
# 6. Evaluate Model
# ---------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))