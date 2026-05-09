import os
from dotenv import load_dotenv
from google import genai

load_dotenv('.env')

gemini_api_key = os.environ.get("GEMINI_API_KEY")
gemini_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

print(f"Gemini API Key Loaded: {bool(gemini_api_key)}")
print(f"Model: {gemini_model_name}")

if gemini_api_key:
    try:
        client = genai.Client(api_key=gemini_api_key)
        response = client.models.generate_content(
            model=gemini_model_name,
            contents="Hello! Are you working?"
        )
        print("Success:", response.text)
    except Exception as e:
        print("Gemini API Error:", e)
else:
    print("API Key not found")
