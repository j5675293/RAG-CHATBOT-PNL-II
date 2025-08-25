import streamlit as st
import os
import time
from datetime import datetime
import pandas as pd

from config import Config
from rag_agent import RAGAgent

# Configuración de la página
st.set_page_config(
    page_title="RAG Chatbot - CVs Estudiantes",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}
.user-message {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
}
.bot-message {
    background-color: #f3e5f5;
    border-left: 4px solid #9c27b0;
}
.confidence-high {
    color: #4caf50;
    font-weight: bold;
}
.confidence-medium {
    color: #ff9800;
    font-weight: bold;
}
.confidence-low {
    color: #f44336;
    font-weight: bold;
}
.stats-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    border: 1px solid #dee2e6;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa el estado de la sesión"""
    if 'rag_agent' not in st.session_state:
        st.session_state.rag_agent = None
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'document_stats' not in st.session_state:
        st.session_state.document_stats = {}

def initialize_rag_agent(use_pinecone: bool = False):
    """Inicializa el agente RAG"""
    try:
        with st.spinner("🤖 Inicializando agente RAG..."):
            agent = RAGAgent(use_pinecone=use_pinecone)
            
            # Intentar cargar índice existente
            if agent.load_existing_index():
                st.session_state.documents_processed = True
            
            st.session_state.rag_agent = agent
            st.success("✅ Agente RAG inicializado correctamente")
            return True
    except Exception as e:
        st.error(f"❌ Error al inicializar RAG: {str(e)}")
        return False

def sidebar():
    """Panel lateral con configuraciones y estadísticas"""
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        
        # Selector de backend vectorial
        use_pinecone = st.checkbox(
            "Usar Pinecone (Cloud)", 
            value=False,
            help="Usa Pinecone para almacenamiento vectorial en la nube. Requiere API key."
        )
        
        # Botón de inicialización
        if st.button("🚀 Inicializar Sistema", type="primary"):
            if initialize_rag_agent(use_pinecone):
                st.rerun()
        
        st.markdown("---")
        
        # Panel de procesamiento de documentos
        st.markdown("## 📄 Documentos")
        
        pdf_directory = st.text_input(
            "Directorio de PDFs", 
            value=Config.PDF_DIRECTORY,
            help="Directorio donde se encuentran los CVs en PDF"
        )
        
        if st.button("🔄 Procesar Documentos"):
            if st.session_state.rag_agent:
                process_documents(pdf_directory)
            else:
                st.error("⚠️ Primero inicializa el sistema")
        
        # Estadísticas
        if st.session_state.rag_agent and st.session_state.documents_processed:
            show_statistics()
        
        st.markdown("---")
        
        # Controles adicionales
        st.markdown("## 🛠️ Controles")
        
        if st.button("🗑️ Limpiar Chat"):
            st.session_state.chat_history = []
            if st.session_state.rag_agent:
                st.session_state.rag_agent.clear_conversation_history()
            st.rerun()
        
        if st.button("💾 Guardar Índice"):
            if st.session_state.rag_agent and not st.session_state.rag_agent.vector_store.use_pinecone:
                st.session_state.rag_agent.vector_store.save_local_index(Config.VECTOR_STORE_PATH)
        
        if st.button("🔥 Limpiar Índice Vectorial"):
            if st.session_state.rag_agent:
                st.session_state.rag_agent.vector_store.clear_index()
                st.session_state.documents_processed = False
                st.success("Índice limpiado")

def process_documents(pdf_directory: str):
    """Procesa los documentos PDF"""
    if not st.session_state.rag_agent:
        st.error("⚠️ Agente RAG no inicializado")
        return
    
    if not os.path.exists(pdf_directory):
        st.error(f"❌ Directorio {pdf_directory} no existe")
        return
    
    with st.spinner("📄 Procesando documentos..."):
        result = st.session_state.rag_agent.process_documents(pdf_directory)
        
        if result["success"]:
            st.success(result["message"])
            st.session_state.documents_processed = True
            st.session_state.document_stats = result["stats"]
            
            # Mostrar estadísticas de procesamiento
            stats = result["stats"]
            st.info(f"""
            📊 **Estadísticas de procesamiento:**
            - **Archivos procesados:** {stats.get('total_files', 0)}
            - **Estudiantes encontrados:** {stats.get('total_students', 0)}
            - **Chunks generados:** {stats.get('total_documents', 0)}
            """)
        else:
            st.error(result["message"])

def show_statistics():
    """Muestra estadísticas del sistema"""
    st.markdown("### 📊 Estadísticas")
    
    try:
        status = st.session_state.rag_agent.get_system_status()
        stats = st.session_state.document_stats
        
        # Estadísticas del vector store
        vector_stats = status["vector_store"]
        st.markdown(f"""
        <div class="stats-card">
        <b>🗄️ Almacén Vectorial</b><br>
        Backend: {vector_stats.get('backend', 'N/A')}<br>
        Vectores: {vector_stats.get('total_vectors', 0)}<br>
        Dimensión: {vector_stats.get('dimension', 0)}
        </div>
        """, unsafe_allow_html=True)
        
        # Estadísticas de documentos
        if stats:
            st.markdown(f"""
            <div class="stats-card">
            <b>📄 Documentos</b><br>
            Estudiantes: {stats.get('total_students', 0)}<br>
            Archivos: {stats.get('total_files', 0)}
            </div>
            """, unsafe_allow_html=True)
            
            # Top habilidades
            if stats.get('top_skills'):
                st.markdown("**🔧 Habilidades más comunes:**")
                for skill, count in stats['top_skills'][:5]:
                    if skill:  # Solo mostrar si no está vacío
                        st.markdown(f"- {skill}: {count}")
    
    except Exception as e:
        st.error(f"Error al mostrar estadísticas: {str(e)}")

def display_chat_message(message: dict, is_user: bool):
    """Muestra un mensaje del chat"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
        <b>👤 Usuario:</b><br>
        {message['content']}
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence_class = f"confidence-{message.get('confidence', 'low')}"
        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(message.get('confidence', 'low'), "⚪")
        
        st.markdown(f"""
        <div class="chat-message bot-message">
        <b>🤖 Asistente:</b><br>
        {message['content']}
        <br><br>
        <small>
        <span class="{confidence_class}">{confidence_emoji} Confianza: {message.get('confidence', 'N/A')}</span> | 
        Tipo: {message.get('query_type', 'N/A')}
        </small>
        </div>
        """, unsafe_allow_html=True)
        
        # Mostrar fuentes si existen
        if message.get('sources'):
            with st.expander("📎 Fuentes utilizadas"):
                for source in message['sources']:
                    st.markdown(f"- {source}")

def show_suggested_queries():
    """Muestra consultas sugeridas"""
    if not st.session_state.rag_agent or not st.session_state.documents_processed:
        return
    
    st.markdown("### 💡 Consultas sugeridas")
    
    suggestions = st.session_state.rag_agent.suggest_queries(st.session_state.document_stats)
    
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions[:6]):
        col = cols[i % 2]
        if col.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
            # Agregar la consulta sugerida al chat
            process_user_query(suggestion)

def process_user_query(query: str):
    """Procesa una consulta del usuario"""
    if not st.session_state.rag_agent:
        st.error("⚠️ Sistema no inicializado")
        return
    
    # Agregar mensaje del usuario al historial
    st.session_state.chat_history.append({
        "content": query,
        "is_user": True,
        "timestamp": datetime.now()
    })
    
    # Procesar consulta con RAG
    with st.spinner("🤔 Pensando..."):
        response = st.session_state.rag_agent.query(query)
    
    # Agregar respuesta del bot al historial
    st.session_state.chat_history.append({
        "content": response["answer"],
        "is_user": False,
        "timestamp": datetime.now(),
        "confidence": response.get("confidence", "low"),
        "query_type": response.get("query_type", "unknown"),
        "sources": response.get("sources", [])
    })

def main():
    """Función principal de la aplicación"""
    # Inicializar estado
    initialize_session_state()
    
    # Header principal
    st.markdown('<h1 class="main-header">🎓 RAG Chatbot - CVs Estudiantes</h1>', unsafe_allow_html=True)
    
    # Panel lateral
    sidebar()
    
    # Contenido principal
    if not st.session_state.rag_agent:
        st.info("👈 **Instrucciones de inicio:**")
        st.markdown("""
        1. **Configura las variables de entorno** en el archivo `.env`
        2. **Inicializa el sistema** usando el botón en el panel lateral
        3. **Procesa los documentos PDF** de CVs
        4. **¡Comienza a hacer preguntas!**
        
        ### 📋 Preparación de archivos:
        - Crea una carpeta llamada `cvs_estudiantes`
        - Coloca los archivos PDF de CVs en esa carpeta
        - Los archivos deben contener información como nombre, email, habilidades, etc.
        """)
        
        # Mostrar ejemplo de estructura de carpetas
        with st.expander("📁 Estructura de carpetas recomendada"):
            st.code("""
proyecto_rag/
├── cvs_estudiantes/
│   ├── cv_juan_perez.pdf
│   ├── cv_maria_garcia.pdf
│   └── cv_carlos_lopez.pdf
├── app.py
├── config.py
├── .env
└── requirements.txt
            """)
        
        return
    
    # Verificar si hay documentos procesados
    if not st.session_state.documents_processed:
        st.warning("⚠️ No hay documentos procesados. Usa el panel lateral para procesar CVs.")
        return
    
    # Mostrar consultas sugeridas
    show_suggested_queries()
    
    st.markdown("---")
    
    # Interface de chat
    st.markdown("### 💬 Chat")
    
    # Área de mensajes del chat
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(message, message["is_user"])
    
    # Input del usuario
    user_input = st.chat_input("Escribe tu pregunta sobre los CVs...")
    
    if user_input:
        process_user_query(user_input)
        st.rerun()
    
    # Panel de información adicional
    with st.expander("ℹ️ Información del sistema"):
        if st.session_state.rag_agent:
            status = st.session_state.rag_agent.get_system_status()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤖 Modelo LLM:**")
                st.code(status["llm_model"])
                
                st.markdown("**🔤 Modelo Embedding:**")
                st.code(status["embedding_model"])
            
            with col2:
                st.markdown("**💾 Vector Store:**")
                st.json(status["vector_store"])
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
    🎓 RAG Chatbot para CVs de Estudiantes | 
    Tecnologías: Streamlit + LangChain + Groq + Pinecone/FAISS
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()