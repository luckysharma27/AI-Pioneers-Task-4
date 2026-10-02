from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

saved = joblib.load("student_score_model.joblib")
model = saved["model"]
features = saved["features"]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            data = {
                "Age": float(request.form["age"]),
                "Study_Hours_Per_Day": float(request.form["study_hours"]),
                "Attendance_Percent": float(request.form["attendance"]),
                "Assignments_Score": float(request.form["assignment"]),
                "Gender_Encoded": 1 if request.form["gender"] == "Female" else 0
            }

            input_df = pd.DataFrame([data])[features]
            prediction = round(float(model.predict(input_df)[0]), 2)
        except Exception:
            error = "Please enter valid values."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
