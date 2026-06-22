import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load Data
df = pd.read_csv("data/advertising.csv")

# Features
X = df[["TV","Radio","Newspaper"]]
y = df["Sales"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R2 Score:", r2_score(y_test, y_pred))

# Sample Prediction
sample = pd.DataFrame({
    "TV":[150],
    "Radio":[25],
    "Newspaper":[20]
})

print("Prediction:", model.predict(sample))