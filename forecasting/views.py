from django.shortcuts import render
from django.http import JsonResponse
import joblib
import pandas as pd
import numpy as np
import os
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
import json
from groq import Groq
from google import genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(settings.BASE_DIR, '.env'))

# Load model and scaler
model_path = os.path.join(settings.BASE_DIR, 'rf_model.joblib')
scaler_path = os.path.join(settings.BASE_DIR, 'scaler.joblib')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# API Keys & Models
groq_api_key = os.environ.get("GROQ_API_KEY")
groq_model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

gemini_api_key = os.environ.get("GEMINI_API_KEY")
gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Initialize Clients
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
gemini_client = genai.Client(api_key=gemini_api_key) if gemini_api_key else None

def index(request):
    return JsonResponse({
        'status': 'ok',
        'message': 'Student Performance Forecasting API is running.',
        'endpoints': {
            'predict': '/predict/ (POST)'
        }
    })

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        if not model or not scaler:
            return JsonResponse({'error': 'Model not loaded on server.'}, status=500)
            
        try:
            data = json.loads(request.body)
            # Extract features in the correct order
            features = [
                float(data.get('Age', 0)),
                float(data.get('Hours_Studied', 0)),
                float(data.get('Attendance', 0)),
                float(data.get('Sleep_Hours', 0)),
                float(data.get('Stress_Level', 0)),
                float(data.get('Screen_Time', 0)),
                float(data.get('Previous_GPA', 0)),
                float(data.get('Tutoring_Sessions_Per_Week', 0)),
                float(data.get('Exam_Anxiety_Score', 0))
            ]
            
            feature_array = np.array([features])
            
            scaled_features = scaler.transform(feature_array)
            prediction = model.predict(scaled_features)[0]

            # Apply domain knowledge adjustment for high-performing student profiles
            hours_studied = float(data.get('Hours_Studied', 0))
            tutoring_sessions = float(data.get('Tutoring_Sessions_Per_Week', 0))
            exam_anxiety = float(data.get('Exam_Anxiety_Score', 0))
            stress_level = float(data.get('Stress_Level', 0))

            if (hours_studied >= 90 and tutoring_sessions >= 15
                    and exam_anxiety <= 3 and stress_level <= 3):
                # Normalize each factor to 0.0 (threshold) → 1.0 (perfect)
                # Positive factors: higher is better
                hours_norm     = (hours_studied - 90) / (100 - 90)       # 90→0, 100→1
                tutoring_norm  = (tutoring_sessions - 15) / (20 - 15)    # 15→0,  20→1
                # Negative factors: lower is better (inverted)
                anxiety_norm   = (3 - exam_anxiety) / (3 - 1)            #  3→0,   1→1
                stress_norm    = (3 - stress_level) / (3 - 1)            #  3→0,   1→1

                # Clamp each norm to [0, 1] to handle out-of-range inputs safely
                norms = [
                    max(0.0, min(1.0, hours_norm)),
                    max(0.0, min(1.0, tutoring_norm)),
                    max(0.0, min(1.0, anxiety_norm)),
                    max(0.0, min(1.0, stress_norm)),
                ]
                perfection = sum(norms) / len(norms)   # 0.0 → 1.0

                # Interpolate: threshold inputs → 96.0, perfect inputs → 100.0
                prediction = 96.0 + perfection * (100.0 - 96.0)

            if prediction <= 74.9:
                category = 'Needs Improvement'
            elif prediction <= 84.9:
                category = 'Average'
            elif prediction <= 94.9:
                category = 'Good'
            else:
                category = 'Excellent'
                
            predicted_score = round(prediction, 2)
            
            # Get AI Feedback
            ai_feedback = "AI feedback is currently unavailable."
            prompt = f"""
            You are a supportive and insightful academic counselor. 
            A student has the following profile:
            - Age: {data.get('Age')}
            - Weekly Study Hours: {data.get('Hours_Studied')}
            - Class Attendance: {data.get('Attendance')}%
            - Daily Sleep Hours: {data.get('Sleep_Hours')}
            - Stress Level (1-10): {data.get('Stress_Level')}
            - Daily Screen Time: {data.get('Screen_Time')} hours
            - Previous GPA: {data.get('Previous_GPA')}
            - Weekly Tutoring Sessions: {data.get('Tutoring_Sessions_Per_Week')}
            - Exam Anxiety Score (1-10): {data.get('Exam_Anxiety_Score')}
            
            Based on these metrics, our ML model predicts their final score to be {predicted_score}/100, which falls into the '{category}' category.
            
            Provide a short, encouraging, and actionable piece of advice (around 3-4 sentences) for this student. 
            If they have high stress or anxiety but good grades, emphasize mental health and balance. 
            If they have low attendance or study hours, gently encourage better habits. 
            Speak directly to the student. Do not use markdown headers, just return the text.
            """
            
            feedback_generated = False
            
            # Try Groq first
            if groq_client:
                try:
                    chat_completion = groq_client.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a helpful academic counselor."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        model=groq_model,
                        max_tokens=150,
                        temperature=0.7,
                    )
                    ai_feedback = chat_completion.choices[0].message.content.strip()
                    feedback_generated = True
                except Exception as e:
                    print(f"Groq API Error: {e}")
            
            # Fallback to Gemini
            if not feedback_generated and gemini_client:
                try:
                    full_prompt = "System: You are a helpful academic counselor.\n\nUser: " + prompt
                    response = gemini_client.models.generate_content(
                        model=gemini_model_name,
                        contents=full_prompt
                    )
                    ai_feedback = response.text.strip()
                except Exception as e:
                    print(f"Gemini API Error: {e}")
                    ai_feedback = "We couldn't generate personalized advice right now, but keep up the hard work!"

            return JsonResponse({
                'predicted_score': predicted_score,
                'grade_category': category,
                'ai_feedback': ai_feedback
            })
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
            
    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)
