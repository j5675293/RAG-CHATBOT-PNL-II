import os
from dotenv import load_dotenv
import sys

print("=== DIAGNÓSTICO DE VARIABLES DE ENTORNO ===")
print(f"Directorio actual: {os.getcwd()}")
print(f"Python version: {sys.version}")

# Verificar si existe el archivo .env
env_file = ".env"
if os.path.exists(env_file):
    print(f"✅ Archivo {env_file} existe")
    
    # Leer contenido del archivo
    with open(env_file, 'r') as f:
        content = f.read()
    print(f"📄 Contenido del archivo .env:")
    print("=" * 40)
    print(repr(content))  # Usar repr para ver caracteres especiales
    print("=" * 40)
else:
    print(f"❌ Archivo {env_file} NO existe")

# Cargar variables de entorno
print("\n=== CARGANDO VARIABLES ===")
result = load_dotenv()
print(f"load_dotenv() resultado: {result}")

# Verificar variable específica
groq_key = os.getenv('GROQ_API_KEY')
print(f"\n=== RESULTADO ===")
print(f"GROQ_API_KEY encontrada: {'✅' if groq_key else '❌'}")
if groq_key:
    print(f"Valor: {groq_key[:20]}...")  # Mostrar solo los primeros 20 caracteres por seguridad
    print(f"Longitud: {len(groq_key)} caracteres")
    print(f"Empieza con 'gsk_': {'✅' if groq_key.startswith('gsk_') else '❌'}")
else:
    print("❌ Variable no encontrada")

# Mostrar todas las variables de entorno que empiecen con GROQ
print(f"\n=== TODAS LAS VARIABLES GROQ ===")
for key, value in os.environ.items():
    if 'GROQ' in key.upper():
        print(f"{key}: {value[:20]}...")
