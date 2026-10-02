# AI Pioneers Internship – Task 4

## AI Project Deployment & Capstone

This project is a simple end-to-end Machine Learning application that predicts a student's final score from basic academic information.

### Workflow
1. Load and clean the student performance dataset.
2. Prepare numerical and categorical features.
3. Train a Random Forest Regression model.
4. Save the trained model using Joblib.
5. Load the saved model in a Flask application.
6. Accept user input through a web form.
7. Return the predicted final score.

### Technologies
- Python
- Pandas
- Scikit-learn
- Joblib
- Flask

### Files
- `train_model.py` – trains and saves the ML model
- `app.py` – Flask prediction API/web application
- `templates/index.html` – simple user interface
- `student_performance_raw.csv` – dataset
- `requirements.txt` – required Python packages

### Run Locally
```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

Open the local Flask address shown in the terminal.
