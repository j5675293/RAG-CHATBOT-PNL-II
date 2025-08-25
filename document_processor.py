import os
import re
from typing import List, Dict, Any
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import streamlit as st

class CVProcessor:
    """Procesador de CVs en PDF con extracción de metadatos específicos"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extrae texto de un archivo PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text
        except Exception as e:
            st.error(f"Error al extraer texto de {pdf_path}: {str(e)}")
            return ""
    
    def extract_cv_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extrae metadatos específicos del CV usando regex"""
        metadata = {
            "filename": filename,
            "student_name": self._extract_name(text),
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "skills": self._extract_skills(text),
            "education": self._extract_education(text),
            "experience": self._extract_experience(text)
        }
        return metadata
    
    def _extract_name(self, text: str) -> str:
        """Extrae el nombre del estudiante"""
        # Patrones comunes para nombres en CVs
        patterns = [
            r"(?:NOMBRE|Name|Nombre):\s*([A-Za-z\s]+)",
            r"^([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",  # Primera línea con nombre
            r"CV\s+(?:de\s+)?([A-Za-z\s]+)",
            r"Curriculum\s+(?:Vitae\s+)?(?:de\s+)?([A-Za-z\s]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name.split()) >= 2 and len(name) > 3:  # Validar que sea un nombre real
                    return name
        
        return "Desconocido"
    
    def _extract_email(self, text: str) -> str:
        """Extrae dirección de email"""
        pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(pattern, text)
        return matches[0] if matches else ""
    
    def _extract_phone(self, text: str) -> str:
        """Extrae número de teléfono"""
        patterns = [
            r'(?:\+?1[-.\s]?)?(?:\(?[0-9]{3}\)?[-.\s]?)?[0-9]{3}[-.\s]?[0-9]{4}',
            r'\+?[\d\s\-\(\)]{10,15}',
            r'(?:Tel|Teléfono|Phone):\s*([\d\s\-\(\)]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        return ""
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extrae habilidades y competencias"""
        skills_section = re.search(
            r'(?:HABILIDADES|SKILLS|COMPETENCIAS|TECNOLOGÍAS).*?(?=\n\n|\n[A-Z]|$)', 
            text, re.IGNORECASE | re.DOTALL
        )
        
        if not skills_section:
            return []
        
        skills_text = skills_section.group()
        
        # Tecnologías comunes
        tech_patterns = [
            r'\b(?:Python|Java|JavaScript|C\+\+|C#|HTML|CSS|SQL|React|Angular|Vue|Node|Django|Flask)\b',
            r'\b(?:Machine Learning|AI|Data Science|Web Development|Frontend|Backend)\b',
            r'\b(?:Git|Docker|Kubernetes|AWS|Azure|Google Cloud)\b'
        ]
        
        skills = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, skills_text, re.IGNORECASE)
            skills.extend(matches)
        
        return list(set(skills))  # Eliminar duplicados
    
    def _extract_education(self, text: str) -> List[str]:
        """Extrae información educativa"""
        education_section = re.search(
            r'(?:EDUCACIÓN|EDUCATION|FORMACIÓN ACADÉMICA).*?(?=\n\n|\n[A-Z]|$)', 
            text, re.IGNORECASE | re.DOTALL
        )
        
        if not education_section:
            return []
        
        education_text = education_section.group()
        
        # Buscar títulos y universidades
        patterns = [
            r'(?:Licenciatura|Ingeniería|Master|Doctorado|Técnico)\s+en\s+[\w\s]+',
            r'Universidad\s+[\w\s]+',
            r'\b(?:Bachelor|Master|PhD|Degree)\s+[\w\s]+'
        ]
        
        education = []
        for pattern in patterns:
            matches = re.findall(pattern, education_text, re.IGNORECASE)
            education.extend(matches)
        
        return education
    
    def _extract_experience(self, text: str) -> List[str]:
        """Extrae experiencia laboral"""
        exp_section = re.search(
            r'(?:EXPERIENCIA|EXPERIENCE|TRABAJO|EMPLOYMENT).*?(?=\n\n|\n[A-Z]|$)', 
            text, re.IGNORECASE | re.DOTALL
        )
        
        if not exp_section:
            return []
        
        exp_text = exp_section.group()
        
        # Buscar posiciones y empresas
        patterns = [
            r'(?:Developer|Desarrollador|Analyst|Engineer|Consultant|Intern)\s*(?:en\s+|at\s+)?[\w\s]+',
            r'(?:Empresa|Company):\s*[\w\s]+',
            r'\b[\w\s]+\s+(?:S\.A\.|Inc\.|LLC|Ltd\.)\b'
        ]
        
        experience = []
        for pattern in patterns:
            matches = re.findall(pattern, exp_text, re.IGNORECASE)
            experience.extend(matches)
        
        return experience
    
    def process_pdf_directory(self, pdf_directory: str) -> List[Document]:
        """Procesa todos los PDFs en un directorio y retorna documentos de LangChain"""
        if not os.path.exists(pdf_directory):
            os.makedirs(pdf_directory)
            st.warning(f"Directorio {pdf_directory} creado. Por favor, agrega archivos PDF de CVs.")
            return []
        
        documents = []
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        
        if not pdf_files:
            st.warning(f"No se encontraron archivos PDF en {pdf_directory}")
            return []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, filename in enumerate(pdf_files):
            status_text.text(f'Procesando: {filename}')
            
            pdf_path = os.path.join(pdf_directory, filename)
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                st.warning(f"No se pudo extraer texto de {filename}")
                continue
            
            # Extraer metadatos del CV
            metadata = self.extract_cv_metadata(text, filename)
            
            # Dividir texto en chunks
            chunks = self.text_splitter.split_text(text)
            
            for j, chunk in enumerate(chunks):
                # Crear metadatos únicos para cada chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": f"{filename}_{j}",
                    "chunk_index": j,
                    "total_chunks": len(chunks),
                    "source_file": pdf_path
                })
                
                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            progress_bar.progress((i + 1) / len(pdf_files))
        
        progress_bar.empty()
        status_text.empty()
        
        return documents
    
    def get_processed_stats(self, documents: List[Document]) -> Dict[str, Any]:
        """Obtiene estadísticas de los documentos procesados"""
        if not documents:
            return {}
        
        unique_files = set(doc.metadata.get('filename', '') for doc in documents)
        unique_students = set(doc.metadata.get('student_name', '') for doc in documents)
        
        all_skills = []
        for doc in documents:
            skills = doc.metadata.get('skills', [])
            all_skills.extend(skills)
        
        skill_counts = {}
        for skill in all_skills:
            skill_counts[skill] = skill_counts.get(skill, 0) + 1
        
        return {
            "total_documents": len(documents),
            "total_files": len(unique_files),
            "total_students": len(unique_students),
            "students_list": list(unique_students),
            "top_skills": sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }