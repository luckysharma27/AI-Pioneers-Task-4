import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv("student_performance_raw.csv").drop_duplicates().reset_index(drop=True)

numeric_cols = [
    "Age", "Study_Hours_Per_Day", "Attendance_Percent",
    "Assignments_Score", "Final_Score"
]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

df["Gender_Encoded"] = df["Gender"].map({"Male": 0, "Female": 1})

features = [
    "Age", "Study_Hours_Per_Day", "Attendance_Percent",
    "Assignments_Score", "Gender_Encoded"
]

X = df[features]
y = df["Final_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(
        n_estimators=100, random_state=42
    ))
])

model.fit(X_train, y_train)

joblib.dump(
    {"model": model, "features": features},
    "student_score_model.joblib"
)

print("Model trained and saved as student_score_model.joblib")
