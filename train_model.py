import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

def train_and_save_model():
    print("Loading data...")
    # Read the dataset
    df = pd.read_csv('student_performance_finalscore.csv')

    # Define features and target
    optimal_features = [
        'Age',
        'Hours_Studied',
        'Attendance',
        'Sleep_Hours',
        'Stress_Level',
        'Screen_Time',
        'Previous_GPA',
        'Tutoring_Sessions_Per_Week',
        'Exam_Anxiety_Score'
    ]
    
    X = df[optimal_features]
    y_continuous = df['Final_Score']

    print("Splitting data...")
    X_train, X_test, y_train_cont, y_test_cont = train_test_split(X, y_continuous, test_size=0.2, random_state=42)

    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_scaled, y_train_cont)

    print("Evaluating model...")
    score = model.score(X_test_scaled, y_test_cont)
    print(f"R^2 Score on Test Set: {score:.4f}")

    print("Saving model and scaler...")
    # Ensure the models directory exists or save in the same directory
    joblib.dump(model, 'rf_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    print("Model and scaler saved successfully.")

if __name__ == "__main__":
    train_and_save_model()
