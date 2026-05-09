import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv('.env')
groq_api_key = os.environ.get("GROQ_API_KEY")

try:
    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        model="llama3-8b-8192",
    )
    print("Success:", chat_completion.choices[0].message.content)
except Exception as e:
    print("Error with llama3-8b-8192:", e)

try:
    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": "Hello"
            }
        ],
        model="llama-3.1-8b-instant",
    )
    print("Success with llama-3.1-8b-instant:", chat_completion.choices[0].message.content)
except Exception as e:
    print("Error with llama-3.1-8b-instant:", e)
