import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------------
# 1. Create Dataset
# -------------------------------

data = {
    "Age": [25, 30, 32, 40, 35],
    "Experience": [2, 5, 7, 5.5, 8],
    "Salary": [3000, 6000, 7000, 8000, 9000]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# -------------------------------
# 2. Define Inputs and Output
# -------------------------------

X = df[["Age", "Experience"]]
y = df["Salary"]

# -------------------------------
# 3. Split Dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

print("\nTraining Data:")
print(X_train)

print("\nTesting Data:")
print(X_test)

# -------------------------------
# 4. Train Linear Regression Model
# -------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# -------------------------------
# 5. Make Predictions
# -------------------------------

predictions = model.predict(X_test)

print("\nPredicted Salary:")
print(predictions)

print("\nActual Salary:")
print(y_test.values)
from sklearn.metrics import r2_score

score = r2_score(y_test, predictions)

print("\nModel Accuracy (R² Score):")
print(score)