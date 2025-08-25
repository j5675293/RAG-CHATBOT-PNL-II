#!/usr/bin/env python3
"""
Script de ejecución rápida para el sistema RAG Chatbot
Ejecuta verificaciones y lanza la aplicación
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def check_environment():
    """Verifica que el entorno esté configurado correctamente"""
    print("🔍 Verificando entorno...")
    
    # Verificar archivo .env
    if not os.path.exists('.env'):
        print("❌ Archivo .env no encontrado")
        print("💡 Ejecuta: python setup.py")
        return False
    
    # Verificar directorio de CVs
    if not os.path.exists('cvs_estudiantes'):
        print("❌ Directorio 'cvs_estudiantes' no encontrado")
        return False
    
    # Verificar archivos PDF
    pdf_files = list(Path('cvs_estudiantes').glob('*.pdf'))
    if not pdf_files:
        print("⚠️ No se encontraron archivos PDF en 'cvs_estudiantes'")
        print("💡 Agrega archivos PDF de CVs para mejores resultados")
    else:
        print(f"✅ Encontrados {len(pdf_files)} archivos PDF")
    
    # Verificar dependencias críticas
    try:
        import streamlit
        import langchain
        import sentence_transformers
        print("✅ Dependencias principales instaladas")
    except ImportError as e:
        print(f"❌ Faltan dependencias: {e}")
        print("💡 Ejecuta: pip install -r requirements.txt")
        return False
    
    return True

def read_env_variables():
    """Lee y verifica las variables de entorno"""
    env_vars = {}
    
    if not os.path.exists('.env'):
        return env_vars
    
    with open('.env', 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                env_vars[key.strip()] = value.strip()
    
    return env_vars

def validate_api_keys():
    """Valida que las API keys estén configuradas"""
    print("🔑 Verificando API keys...")
    
    env_vars = read_env_variables()
    
    # Verificar GROQ API key (obligatoria)
    groq_key = env_vars.get('GROQ_API_KEY', '')
    if not groq_key or groq_key == 'your_groq_api_key_here':
        print("❌ GROQ API key no configurada")
        print("💡 Edita .env y agrega tu GROQ API key")
        return False
    else:
        print("✅ GROQ API key configurada")
    
    # Verificar Pinecone (opcional)
    pinecone_key = env_vars.get('PINECONE_API_KEY', '')
    if pinecone_key and pinecone_key != 'your_pinecone_api_key_here':
        print("✅ Pinecone API key configurada")
    else:
        print("ℹ️ Pinecone no configurado (usará FAISS local)")
    
    return True

def launch_streamlit():
    """Lanza la aplicación Streamlit"""
    print("🚀 Iniciando aplicación Streamlit...")
    
    try:
        # Lanzar Streamlit
        cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
        
        print("📡 La aplicación se abrirá en: http://localhost:8501")
        print("⏹️ Presiona Ctrl+C para detener la aplicación")
        
        # Abrir navegador automáticamente
        try:
            webbrowser.open("http://localhost:8501")
        except:
            pass
        
        # Ejecutar Streamlit
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Aplicación detenida por el usuario")
    except FileNotFoundError:
        print("❌ Streamlit no encontrado")
        print("💡 Instala con: pip install streamlit")
    except Exception as e:
        print(f"❌ Error al lanzar aplicación: {e}")

def show_quick_help():
    """Muestra ayuda rápida"""
    print("\n" + "="*60)
    print("🤖 RAG CHATBOT - GUÍA RÁPIDA")
    print("="*60)
    
    print("\n📋 UNA VEZ EN LA APLICACIÓN:")
    print("1. 🚀 Inicializar Sistema (panel lateral)")
    print("2. 🔄 Procesar Documentos (si hay PDFs nuevos)")
    print("3. 💬 Hacer preguntas sobre los CVs")
    
    print("\n💡 EJEMPLOS DE PREGUNTAS:")
    print('• "¿Qué estudiantes tienes?"')
    print('• "Busca estudiantes con Python"')
    print('• "Quién tiene experiencia en JavaScript?"')
    print('• "Información de contacto"')
    
    print("\n🔧 CONFIGURACIÓN:")
    print("• Edita .env para cambiar API keys")
    print("• Agrega PDFs en cvs_estudiantes/")
    print("• Usa Pinecone o FAISS (local)")
    
    print("\n" + "="*60)

def show_system_info():
    """Muestra información del sistema"""
    print("\n📊 INFORMACIÓN DEL SISTEMA:")
    
    # Python version
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Check dependencies
    deps = ['streamlit', 'langchain', 'sentence_transformers', 'pinecone', 'groq']
    for dep in deps:
        try:
            module = __import__(dep.replace('-', '_'))
            version = getattr(module, '__version__', 'N/A')
            print(f"📦 {dep}: {version}")
        except ImportError:
            print(f"❌ {dep}: No instalado")
    
    # Environment info
    env_vars = read_env_variables()
    print(f"\n🔑 Variables configuradas: {len(env_vars)}")
    
    # Files info
    pdf_count = len(list(Path('cvs_estudiantes').glob('*.pdf'))) if os.path.exists('cvs_estudiantes') else 0
    print(f"📄 PDFs encontrados: {pdf_count}")
    
    # Vector store info
    if os.path.exists('vector_store'):
        faiss_files = list(Path('vector_store').glob('*.bin'))
        print(f"💾 Índices FAISS: {len(faiss_files)}")

def main():
    """Función principal"""
    print("🎓 RAG Chatbot - Launcher")
    print("="*40)
    
    # Mostrar información del sistema
    show_system_info()
    
    # Verificaciones
    if not check_environment():
        print("\n❌ Entorno no configurado correctamente")
        print("💡 Ejecuta primero: python setup.py")
        return
    
    if not validate_api_keys():
        print("\n❌ API keys no configuradas")
        return
    
    print("\n✅ Sistema listo para ejecutar")
    
    # Mostrar ayuda
    show_quick_help()
    
    # Preguntar si continuar
    try:
        response = input("\n¿Iniciar aplicación? (y/n): ").lower()
        if response in ['y', 'yes', 'sí', 's', '']:
            launch_streamlit()
        else:
            print("👋 ¡Hasta luego!")
    except KeyboardInterrupt:
        print("\n👋 ¡Hasta luego!")

if __name__ == "__main__":
    main()