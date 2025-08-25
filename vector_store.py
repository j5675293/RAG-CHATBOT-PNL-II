import os
import time
from typing import List, Dict, Any, Optional
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from langchain.schema import Document
import pinecone
from config import Config

class VectorStoreManager:
    """Gestor del almacén vectorial con soporte para FAISS local y Pinecone"""
    
    def __init__(self, use_pinecone: bool = False):
        self.use_pinecone = use_pinecone
        self.embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        
        # Almacenamiento local
        self.faiss_index = None
        self.documents = []
        self.document_metadata = []
        
        # Pinecone
        self.pinecone_index = None
        
        if use_pinecone:
            self._initialize_pinecone()
        else:
            self._initialize_faiss()
    
    def _initialize_pinecone(self):
        """Inicializa la conexión con Pinecone"""
        try:
            pinecone.init(
                api_key=Config.PINECONE_API_KEY,
                environment=Config.PINECONE_ENVIRONMENT
            )
            
            # Crear índice si no existe
            if Config.PINECONE_INDEX_NAME not in pinecone.list_indexes():
                pinecone.create_index(
                    Config.PINECONE_INDEX_NAME,
                    dimension=self.embedding_dimension,
                    metric="cosine"
                )
                time.sleep(10)  # Esperar a que el índice esté listo
            
            self.pinecone_index = pinecone.Index(Config.PINECONE_INDEX_NAME)
            st.success("✅ Conectado a Pinecone correctamente")
            
        except Exception as e:
            st.error(f"❌ Error al conectar con Pinecone: {str(e)}")
            st.info("🔄 Usando almacenamiento local FAISS como respaldo")
            self.use_pinecone = False
            self._initialize_faiss()
    
    def _initialize_faiss(self):
        """Inicializa el índice FAISS local"""
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner Product para cosine similarity
        st.info("📁 Usando almacenamiento vectorial local (FAISS)")
    
    def add_documents(self, documents: List[Document]):
        """Agrega documentos al almacén vectorial"""
        if not documents:
            return
        
        # Extraer textos para embedding
        texts = [doc.page_content for doc in documents]
        
        # Generar embeddings
        st.info("🔄 Generando embeddings...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        if self.use_pinecone:
            self._add_to_pinecone(documents, embeddings)
        else:
            self._add_to_faiss(documents, embeddings)
        
        st.success(f"✅ {len(documents)} documentos agregados al almacén vectorial")
    
    def _add_to_pinecone(self, documents: List[Document], embeddings: np.ndarray):
        """Agrega documentos a Pinecone"""
        vectors_to_upsert = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector_id = f"doc_{int(time.time())}_{i}"
            metadata = {
                "text": doc.page_content[:1000],  # Limitar texto para metadatos
                "filename": doc.metadata.get('filename', ''),
                "student_name": doc.metadata.get('student_name', ''),
                "chunk_id": doc.metadata.get('chunk_id', ''),
                "skills": str(doc.metadata.get('skills', [])),
                "education": str(doc.metadata.get('education', [])),
                "experience": str(doc.metadata.get('experience', []))
            }
            
            vectors_to_upsert.append((vector_id, embedding.tolist(), metadata))
        
        # Upsert en batches
        batch_size = 100
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i+batch_size]
            self.pinecone_index.upsert(vectors=batch)
    
    def _add_to_faiss(self, documents: List[Document], embeddings: np.ndarray):
        """Agrega documentos a FAISS"""
        # Normalizar embeddings para cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Agregar al índice FAISS
        self.faiss_index.add(embeddings.astype('float32'))
        
        # Guardar documentos y metadatos
        self.documents.extend(documents)
        self.document_metadata.extend([doc.metadata for doc in documents])
    
    def search(self, query: str, k: int = 5, filters: Dict[str, Any] = None) -> List[Dict]:
        """Busca documentos similares a la consulta"""
        # Generar embedding de la consulta
        query_embedding = self.embedding_model.encode([query])
        
        if self.use_pinecone:
            return self._search_pinecone(query_embedding[0], k, filters)
        else:
            return self._search_faiss(query_embedding[0], k, filters)
    
    def _search_pinecone(self, query_embedding: np.ndarray, k: int, filters: Dict[str, Any]) -> List[Dict]:
        """Busca en Pinecone"""
        # Construir filtros de Pinecone
        pinecone_filters = {}
        if filters:
            if filters.get('student_name'):
                pinecone_filters['student_name'] = {'$eq': filters['student_name']}
            if filters.get('skill'):
                pinecone_filters['skills'] = {'$regex': f".*{filters['skill']}.*"}
        
        # Realizar búsqueda
        results = self.pinecone_index.query(
            vector=query_embedding.tolist(),
            top_k=k,
            include_metadata=True,
            filter=pinecone_filters if pinecone_filters else None
        )
        
        # Formatear resultados
        formatted_results = []
        for match in results['matches']:
            formatted_results.append({
                'content': match['metadata'].get('text', ''),
                'metadata': match['metadata'],
                'score': match['score']
            })
        
        return formatted_results
    
    def _search_faiss(self, query_embedding: np.ndarray, k: int, filters: Dict[str, Any]) -> List[Dict]:
        """Busca en FAISS local"""
        if self.faiss_index.ntotal == 0:
            return []
        
        # Normalizar embedding de consulta
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Buscar en FAISS
        scores, indices = self.faiss_index.search(query_embedding, min(k * 2, self.faiss_index.ntotal))
        
        # Filtrar y formatear resultados
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Índice inválido
                continue
            
            doc = self.documents[idx]
            metadata = self.document_metadata[idx]
            
            # Aplicar filtros
            if self._apply_filters(metadata, filters):
                results.append({
                    'content': doc.page_content,
                    'metadata': metadata,
                    'score': float(score)
                })
        
        # Limitar resultados
        return results[:k]
    
    def _apply_filters(self, metadata: Dict, filters: Dict[str, Any]) -> bool:
        """Aplica filtros a los metadatos"""
        if not filters:
            return True
        
        if filters.get('student_name'):
            if filters['student_name'].lower() not in metadata.get('student_name', '').lower():
                return False
        
        if filters.get('skill'):
            skills_str = ' '.join(metadata.get('skills', []))
            if filters['skill'].lower() not in skills_str.lower():
                return False
        
        return True
    
    def save_local_index(self, path: str):
        """Guarda el índice FAISS local"""
        if not self.use_pinecone and self.faiss_index:
            os.makedirs(path, exist_ok=True)
            
            # Guardar índice FAISS
            faiss.write_index(self.faiss_index, os.path.join(path, 'faiss_index.bin'))
            
            # Guardar documentos y metadatos
            with open(os.path.join(path, 'documents.pkl'), 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(os.path.join(path, 'metadata.pkl'), 'wb') as f:
                pickle.dump(self.document_metadata, f)
            
            st.success(f"✅ Índice guardado en {path}")
    
    def load_local_index(self, path: str) -> bool:
        """Carga el índice FAISS local"""
        if self.use_pinecone:
            return False
        
        try:
            # Cargar índice FAISS
            index_path = os.path.join(path, 'faiss_index.bin')
            if os.path.exists(index_path):
                self.faiss_index = faiss.read_index(index_path)
            else:
                return False
            
            # Cargar documentos
            docs_path = os.path.join(path, 'documents.pkl')
            if os.path.exists(docs_path):
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
            else:
                return False
            
            # Cargar metadatos
            metadata_path = os.path.join(path, 'metadata.pkl')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.document_metadata = pickle.load(f)
            else:
                return False
            
            st.success(f"✅ Índice cargado desde {path}")
            return True
            
        except Exception as e:
            st.error(f"❌ Error al cargar índice: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del almacén vectorial"""
        if self.use_pinecone:
            try:
                stats = self.pinecone_index.describe_index_stats()
                return {
                    "backend": "Pinecone",
                    "total_vectors": stats.get('total_vector_count', 0),
                    "dimension": self.embedding_dimension
                }
            except:
                return {"backend": "Pinecone", "status": "Error"}
        else:
            return {
                "backend": "FAISS Local",
                "total_vectors": self.faiss_index.ntotal if self.faiss_index else 0,
                "total_documents": len(self.documents),
                "dimension": self.embedding_dimension
            }
    
    def clear_index(self):
        """Limpia el índice vectorial"""
        if self.use_pinecone:
            try:
                # Eliminar todos los vectores de Pinecone
                self.pinecone_index.delete(delete_all=True)
                st.success("✅ Índice Pinecone limpiado")
            except Exception as e:
                st.error(f"❌ Error al limpiar Pinecone: {str(e)}")
        else:
            # Reinicializar FAISS
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)
            self.documents = []
            self.document_metadata = []
            st.success("✅ Índice FAISS local limpiado")