import os
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east-1-aws")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "student-cv-index")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Modelo de embedding
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Configuraciones de RAG - Ajustadas para mejor rendimiento
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RETRIEVAL = 3  # Reducido para mejor precisión
    SIMILARITY_THRESHOLD = 0.7  # Umbral de similitud más permisivo
    
    # Directorios
    PDF_DIRECTORY = "cvs_estudiantes"
    VECTOR_STORE_PATH = "vector_store"
    
    # Configuración del modelo de chat
    MODEL_NAME = "llama3-70b-8192"
    TEMPERATURE = 0.1
    MAX_TOKENS = 1024
    
    @classmethod
    def validate_config(cls):
        """Valida que todas las configuraciones necesarias estén presentes"""
        required_keys = [
            "GROQ_API_KEY",
            "PINECONE_API_KEY"
        ]
        
        missing_keys = []
        for key in required_keys:
            value = getattr(cls, key)
            if not value or value.strip() == "":
                missing_keys.append(key)
        
        if missing_keys:
            error_msg = f"Faltan las siguientes variables de entorno: {', '.join(missing_keys)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("✅ Configuración validada correctamente")
        return True
    
    @classmethod
    def print_config(cls):
        """Imprime la configuración actual (sin mostrar API keys completas)"""
        print("🔧 Configuración actual:")
        print(f"  - Modelo: {cls.MODEL_NAME}")
        print(f"  - Top K: {cls.TOP_K_RETRIEVAL}")
        print(f"  - Umbral similitud: {cls.SIMILARITY_THRESHOLD}")
        print(f"  - GROQ API: {'✅' if cls.GROQ_API_KEY else '❌'}")
        print(f"  - Pinecone API: {'✅' if cls.PINECONE_API_KEY else '❌'}")

# Validar configuración al importar
try:
    Config.validate_config()
except ValueError as e:
    logger.error(f"❌ Error de configuración: {e}")
    raise