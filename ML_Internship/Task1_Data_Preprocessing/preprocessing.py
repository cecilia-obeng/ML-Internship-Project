# Task 1: Data Preprocessing

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------------
# 1. Create Dataset
# -----------------------------------

data = {
    "Age": [25, 30, 32, 40, 35],
    "Experience": [2, 5, 7, 5.5, 8],
    "Salary": [3000, 6000, 7000, 8000, 9000],
    "Gender": ["Male", "Female", "Female", "Male", "Female"]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)
print("\n")

# -----------------------------------
# 2. Encode Categorical Data
# -----------------------------------

df = pd.get_dummies(df, columns=["Gender"], drop_first=True)

print("Dataset After Encoding:")
print(df)
print("\n")

# -----------------------------------
# 3. Separate Features and Target
# -----------------------------------

X = df.drop("Salary", axis=1)
y = df["Salary"]

# -----------------------------------
# 4. Split Dataset
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Data:")
print(X_train)
print("\n")

print("Testing Data:")
print(X_test)
print("\n")

# -----------------------------------
# 5. Feature Scaling
# -----------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Scaled Training Data:")
print(X_train_scaled)
print("\n")

print("Scaled Testing Data:")
print(X_test_scaled)