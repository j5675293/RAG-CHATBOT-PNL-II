from typing import List, Dict, Any, Optional
import streamlit as st
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from config import Config
from decision_engine import DecisionEngine, QueryType, ResponseTemplates
from vector_store import VectorStoreManager
from document_processor import CVProcessor

class RAGAgent:
    """Agente RAG principal que integra todos los componentes"""
    
    def __init__(self, use_pinecone: bool = False):
        # Validar configuración
        Config.validate_config()
        
        # Inicializar componentes
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )
        
        self.decision_engine = DecisionEngine()
        self.vector_store = VectorStoreManager(use_pinecone=use_pinecone)
        self.document_processor = CVProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        # Plantillas de prompts
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question", "query_type"],
            template=self._get_system_prompt()
        )
        
        # Estado de la conversación
        self.conversation_history = []
    
    def _get_system_prompt(self) -> str:
        """Obtiene el prompt del sistema para el LLM"""
        return """Eres un asistente especializado en información de CVs de estudiantes. 
Tu función es ayudar a los usuarios a encontrar información específica sobre estudiantes basándote en los CVs almacenados.

CONTEXTO RECUPERADO:
{context}

TIPO DE CONSULTA: {query_type}

CONSULTA DEL USUARIO: {question}

INSTRUCCIONES:
1. Utiliza ÚNICAMENTE la información proporcionada en el contexto para responder
2. Si el contexto no contiene información suficiente, indícalo claramente
3. Sé específico y detallado en tus respuestas
4. Mantén un tono profesional y útil
5. Si mencionas nombres de estudiantes, asegúrate de que estén en el contexto
6. Para consultas sobre habilidades, experiencia o educación, organiza la información de manera clara

RESPUESTA:"""
    
    def process_documents(self, pdf_directory: str) -> Dict[str, Any]:
        """Procesa documentos PDF y los agrega al almacén vectorial"""
        st.info("🔄 Procesando documentos PDF...")
        
        # Procesar PDFs
        documents = self.document_processor.process_pdf_directory(pdf_directory)
        
        if not documents:
            return {"success": False, "message": "No se encontraron documentos para procesar"}
        
        # Agregar al almacén vectorial
        self.vector_store.add_documents(documents)
        
        # Guardar índice local si aplica
        if not self.vector_store.use_pinecone:
            self.vector_store.save_local_index(Config.VECTOR_STORE_PATH)
        
        # Obtener estadísticas
        stats = self.document_processor.get_processed_stats(documents)
        
        return {
            "success": True,
            "stats": stats,
            "message": f"Procesados {len(documents)} chunks de {stats.get('total_files', 0)} archivos"
        }
    
    def query(self, user_question: str) -> Dict[str, Any]:
        """Procesa una consulta del usuario usando RAG"""
        
        # Análisis de la consulta usando el motor de decisiones
        query_type, extracted_info = self.decision_engine.analyze_query(user_question)
        
        # Obtener estrategia de búsqueda
        search_strategy = self.decision_engine.get_search_strategy(query_type, extracted_info)
        
        # Determinar si usar RAG
        if not self.decision_engine.should_use_rag(query_type):
            return self._handle_non_rag_query(query_type, user_question)
        
        # Realizar búsqueda vectorial
        search_results = self.vector_store.search(
            query=user_question,
            k=search_strategy["max_results"],
            filters=search_strategy.get("search_filters", {})
        )
        
        if not search_results:
            return {
                "answer": "Lo siento, no encontré información relevante para tu consulta en los CVs almacenados.",
                "query_type": query_type.value,
                "sources": [],
                "confidence": "low"
            }
        
        # Construir contexto
        context = self._build_context(search_results)
        
        # Generar respuesta usando LLM
        response = self._generate_response(user_question, context, query_type, search_strategy)
        
        # Guardar en historial
        self.conversation_history.append({
            "question": user_question,
            "answer": response["answer"],
            "query_type": query_type.value,
            "timestamp": st.session_state.get('timestamp', '')
        })
        
        return response
    
    def _handle_non_rag_query(self, query_type: QueryType, question: str) -> Dict[str, Any]:
        """Maneja consultas que no requieren RAG"""
        if query_type == QueryType.GREETING:
            template = ResponseTemplates.get_template("greeting")
            return {
                "answer": template,
                "query_type": query_type.value,
                "sources": [],
                "confidence": "high"
            }
        
        return {
            "answer": "No pude entender tu consulta. ¿Podrías reformularla?",
            "query_type": query_type.value,
            "sources": [],
            "confidence": "low"
        }
    
    def _build_context(self, search_results: List[Dict]) -> str:
        """Construye el contexto a partir de los resultados de búsqueda"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            content = result['content'][:500]  # Limitar longitud
            metadata = result['metadata']
            
            context_part = f"""
DOCUMENTO {i}:
Archivo: {metadata.get('filename', 'Desconocido')}
Estudiante: {metadata.get('student_name', 'Desconocido')}
Contenido: {content}
Puntuación: {result.get('score', 0):.3f}
---"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_response(self, question: str, context: str, query_type: QueryType, 
                          search_strategy: Dict) -> Dict[str, Any]:
        """Genera respuesta usando el LLM"""
        
        try:
            # Crear prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question,
                query_type=query_type.value
            )
            
            # Generar respuesta
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Aplicar template si es necesario
            template_type = search_strategy.get("response_template", "default")
            if template_type != "default":
                template = ResponseTemplates.get_template(template_type)
                if "{context}" in template:
                    answer = template.format(context=answer, **search_strategy.get("search_filters", {}))
            
            # Extraer fuentes
            sources = self._extract_sources_from_context(context)
            
            # Calcular confianza
            confidence = self._calculate_confidence(context, answer)
            
            return {
                "answer": answer,
                "query_type": query_type.value,
                "sources": sources,
                "confidence": confidence,
                "context_used": len(context) > 0
            }
            
        except Exception as e:
            st.error(f"Error al generar respuesta: {str(e)}")
            return {
                "answer": "Lo siento, ocurrió un error al generar la respuesta. Por favor, intenta de nuevo.",
                "query_type": query_type.value,
                "sources": [],
                "confidence": "error"
            }
    
    def _extract_sources_from_context(self, context: str) -> List[str]:
        """Extrae fuentes del contexto"""
        import re
        sources = []
        
        # Buscar nombres de archivos
        file_matches = re.findall(r'Archivo:\s*([^\n]+)', context)
        sources.extend(file_matches)
        
        return list(set(sources))  # Eliminar duplicados
    
    def _calculate_confidence(self, context: str, answer: str) -> str:
        """Calcula el nivel de confianza de la respuesta"""
        if not context:
            return "low"
        
        if len(context) > 1000 and "Desconocido" not in answer:
            return "high"
        elif len(context) > 500:
            return "medium"
        else:
            return "low"
    
    def get_conversation_history(self) -> List[Dict]:
        """Obtiene el historial de conversación"""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Limpia el historial de conversación"""
        self.conversation_history = []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene el estado del sistema"""
        vector_stats = self.vector_store.get_stats()
        
        return {
            "vector_store": vector_stats,
            "conversation_history": len(self.conversation_history),
            "llm_model": Config.MODEL_NAME,
            "embedding_model": Config.EMBEDDING_MODEL
        }
    
    def load_existing_index(self) -> bool:
        """Carga índice existente si está disponible"""
        if not self.vector_store.use_pinecone:
            return self.vector_store.load_local_index(Config.VECTOR_STORE_PATH)
        return True
    
    def suggest_queries(self, documents_stats: Dict = None) -> List[str]:
        """Sugiere consultas basadas en los documentos disponibles"""
        base_suggestions = [
            "¿Qué estudiantes tienes en la base de datos?",
            "Busca estudiantes con experiencia en Python",
            "¿Quién tiene conocimientos en Machine Learning?",
            "Muéstrame información sobre experiencia laboral",
            "¿Qué estudiantes han estudiado Ingeniería?"
        ]
        
        if documents_stats and documents_stats.get("students_list"):
            student_suggestions = [
                f"Busca información de {name}" 
                for name in documents_stats["students_list"][:3] 
                if name != "Desconocido"
            ]
            base_suggestions.extend(student_suggestions)
        
        if documents_stats and documents_stats.get("top_skills"):
            skill_suggestions = [
                f"¿Quién tiene experiencia en {skill[0]}?" 
                for skill in documents_stats["top_skills"][:2]
            ]
            base_suggestions.extend(skill_suggestions)
        
        return base_suggestions