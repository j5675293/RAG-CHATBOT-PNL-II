from dotenv import load_dotenv
import os

load_dotenv()
print("GROQ_API_KEY:", os.getenv("GROQ_API_KEY"))