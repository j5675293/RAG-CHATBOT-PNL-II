from dotenv import load_dotenv
import os

load_dotenv()
model_name = os.getenv("MODEL_NAME")
print("✅ Modelo activo desde .env:", model_name)