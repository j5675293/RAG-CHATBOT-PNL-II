#!/usr/bin/env python3
"""
Script de configuración inicial para el sistema RAG Chatbot
Automatiza la creación de directorios y verificación de dependencias
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def create_directory_structure():
    """Crea la estructura de directorios necesaria"""
    directories = [
        "cvs_estudiantes",
        "vector_store",
        "logs"
    ]
    
    print("📁 Creando estructura de directorios...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Directorio creado: {directory}")
        else:
            print(f"ℹ️ Directorio ya existe: {directory}")

def check_python_version():
    """Verifica la versión de Python"""
    print("🐍 Verificando versión de Python...")
    
    if sys.version_info < (3, 8):
        print("❌ Se requiere Python 3.8 o superior")
        print(f"Versión actual: {sys.version}")
        return False
    
    print(f"✅ Versión de Python: {sys.version}")
    return True

def install_dependencies():
    """Instala las dependencias del proyecto"""
    print("📦 Instalando dependencias...")
    
    try:
        # Actualizar pip
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Instalar dependencias desde requirements.txt
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        
        print("✅ Dependencias instaladas correctamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error al instalar dependencias: {e}")
        return False

def create_env_file():
    """Crea archivo .env con plantilla"""
    if os.path.exists('.env'):
        print("ℹ️ Archivo .env ya existe")
        return
    
    print("📝 Creando archivo .env...")
    
    env_template = """# Variables de entorno para RAG Chatbot
# Reemplaza los valores con tus API keys reales

# GROQ API (Requerido)
GROQ_API_KEY=your_groq_api_key_here

# PINECONE API (Opcional - solo si usas Pinecone)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1-aws

# OPENAI API (Opcional - para embeddings alternativos)
OPENAI_API_KEY=your_openai_api_key_here

# Configuración del índice
PINECONE_INDEX_NAME=student-cv-index
"""
    
    with open('.env', 'w') as f:
        f.write(env_template)
    
    print("✅ Archivo .env creado")
    print("⚠️ IMPORTANTE: Edita el archivo .env con tus API keys reales")

def create_sample_data():
    """Crea datos de ejemplo para testing"""
    sample_dir = "cvs_estudiantes"
    sample_file = os.path.join(sample_dir, "README.md")
    
    if os.path.exists(sample_file):
        return
    
    print("📄 Creando documentación de ejemplo...")
    
    readme_content = """# CVs de Estudiantes

Este directorio debe contener los archivos PDF de los CVs de estudiantes.

## Estructura recomendada de archivos:
- cv_nombre_apellido.pdf
- cv_estudiante_001.pdf
- etc.

## Formato recomendado para CVs:

Los CVs deben incluir las siguientes secciones para mejor extracción:
- **Nombre completo**
- **Información de contacto** (email, teléfono)
- **Habilidades técnicas**
- **Experiencia laboral**
- **Educación**

## Ejemplo de contenido:

```
JUAN PÉREZ GARCÍA
Email: juan.perez@email.com
Teléfono: +1234567890

HABILIDADES:
- Python
- JavaScript
- Machine Learning
- SQL

EXPERIENCIA:
- Desarrollador Junior en TechCorp (2023)
- Intern en StartupXYZ (2022)

EDUCACIÓN:
- Ingeniería en Sistemas - Universidad ABC (2024)
```

Una vez que agregues archivos PDF a este directorio, usa la aplicación Streamlit para procesarlos.
"""
    
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("✅ Documentación de ejemplo creada")

def check_api_keys():
    """Verifica si las API keys están configuradas"""
    print("🔑 Verificando configuración de API keys...")
    
    if not os.path.exists('.env'):
        print("⚠️ Archivo .env no encontrado")
        return False
    
    with open('.env', 'r') as f:
        env_content = f.read()
    
    if 'your_groq_api_key_here' in env_content:
        print("⚠️ GROQ API key no configurada en .env")
        return False
    
    print("✅ Archivo .env configurado (verifica que las keys sean válidas)")
    return True

def run_tests():
    """Ejecuta tests básicos del sistema"""
    print("🧪 Ejecutando tests básicos...")
    
    try:
        # Test de importación de módulos principales
        from config import Config
        from rag_agent import RAGAgent
        from decision_engine import DecisionEngine
        from document_processor import CVProcessor
        from vector_store import VectorStoreManager
        
        print("✅ Todos los módulos se importan correctamente")
        
        # Test de configuración
        try:
            Config.validate_config()
            print("✅ Configuración válida")
        except ValueError as e:
            print(f"⚠️ Configuración incompleta: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False

def print_next_steps():
    """Imprime los próximos pasos para el usuario"""
    print("\n" + "="*60)
    print("🎉 CONFIGURACIÓN COMPLETADA")
    print("="*60)
    
    print("\n📋 PRÓXIMOS PASOS:")
    print("\n1. 🔑 Configurar API Keys:")
    print("   - Edita el archivo .env con tus API keys reales")
    print("   - GROQ API key es OBLIGATORIA")
    print("   - Pinecone API key es opcional (usa FAISS local si no tienes)")
    
    print("\n2. 📄 Agregar CVs:")
    print("   - Coloca archivos PDF de CVs en el directorio 'cvs_estudiantes/'")
    print("   - Usa nombres descriptivos como 'cv_juan_perez.pdf'")
    
    print("\n3. 🚀 Ejecutar la aplicación:")
    print("   streamlit run app.py")
    
    print("\n4. 🖥️ Usar la aplicación:")
    print("   - Abre http://localhost:8501 en tu navegador")
    print("   - Inicializa el sistema desde el panel lateral")
    print("   - Procesa los documentos PDF")
    print("   - ¡Comienza a hacer preguntas!")
    
    print("\n" + "="*60)

def main():
    """Función principal del script de configuración"""
    print("🤖 RAG Chatbot - Script de Configuración Inicial")
    print("="*60)
    
    # Verificaciones y configuración
    if not check_python_version():
        sys.exit(1)
    
    create_directory_structure()
    create_env_file()
    create_sample_data()
    
    if install_dependencies():
        print("✅ Dependencias instaladas")
    else:
        print("❌ Error en instalación de dependencias")
        sys.exit(1)
    
    # Tests y verificaciones
    run_tests()
    check_api_keys()
    
    # Mostrar próximos pasos
    print_next_steps()

if __name__ == "__main__":
    main()