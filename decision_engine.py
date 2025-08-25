import re
from typing import Dict, List, Tuple
from enum import Enum

class QueryType(Enum):
    STUDENT_SEARCH = "student_search"
    SKILL_QUERY = "skill_query"
    EXPERIENCE_QUERY = "experience_query"
    EDUCATION_QUERY = "education_query"
    CONTACT_QUERY = "contact_query"
    GENERAL_INFO = "general_info"
    GREETING = "greeting"
    UNKNOWN = "unknown"

class DecisionEngine:
    """
    Motor de decisiones que utiliza conditional edges para determinar
    el tipo de consulta y la estrategia de respuesta apropiada
    """
    
    def __init__(self):
        # Patrones regex para diferentes tipos de consulta
        self.patterns = {
            QueryType.STUDENT_SEARCH: [
                r"busca[r]?\s+estudiante[s]?\s+(?:llamado[s]?|de nombre|que se llam[ae])\s+(\w+)",
                r"estudiante[s]?\s+(?:con nombre|llamado[s]?)\s+(\w+)",
                r"quien es\s+(\w+)",
                r"información de\s+(\w+)",
                r"datos de[l]?\s+estudiante\s+(\w+)"
            ],
            QueryType.SKILL_QUERY: [
                r"(?:habilidades?|skills?|competencias?|conocimientos?)",
                r"(?:sabe|conoce|domina)\s+(?:de\s+)?(\w+)",
                r"experiencia en\s+(\w+)",
                r"tecnolog[íi]as?\s+(?:que\s+)?(?:maneja|conoce|domina)"
            ],
            QueryType.EXPERIENCE_QUERY: [
                r"experiencia\s+laboral",
                r"(?:trabajos?|empleos?)\s+(?:anteriores?|previos?)",
                r"(?:dónde|donde)\s+(?:ha\s+)?trabajado",
                r"empresas?\s+(?:donde\s+ha\s+trabajado|en\s+las\s+que\s+trabajó)"
            ],
            QueryType.EDUCATION_QUERY: [
                r"(?:educación|estudios?|formación)\s+(?:académica?)?",
                r"(?:universidad|carrera|titulo|grado)",
                r"(?:dónde|donde)\s+(?:estudió|estudia)",
                r"(?:qué|que)\s+(?:estudió|estudia|carrera)"
            ],
            QueryType.CONTACT_QUERY: [
                r"(?:contacto|teléfono|email|correo|dirección)",
                r"(?:cómo|como)\s+(?:contactar|comunicarse)",
                r"datos\s+de\s+contacto"
            ],
            QueryType.GREETING: [
                r"^(?:hola|buenos?\s+días?|buenas?\s+tardes?|buenas?\s+noches?)",
                r"^(?:saludos?|hi|hello)",
                r"^(?:qué\s+tal|como\s+estas?|como\s+está)"
            ]
        }
    
    def analyze_query(self, query: str) -> Tuple[QueryType, Dict]:
        """
        Analiza la consulta y determina el tipo y extrae información relevante
        usando conditional edges
        """
        query_lower = query.lower().strip()
        extracted_info = {}
        
        # Conditional Edge 1: Verificar saludos
        if self._match_patterns(query_lower, QueryType.GREETING):
            return QueryType.GREETING, {}
        
        # Conditional Edge 2: Búsqueda específica de estudiante
        student_match = self._extract_student_name(query_lower)
        if student_match:
            extracted_info["student_name"] = student_match
            return QueryType.STUDENT_SEARCH, extracted_info
        
        # Conditional Edge 3: Consultas sobre habilidades
        if self._match_patterns(query_lower, QueryType.SKILL_QUERY):
            skill_match = self._extract_skill(query_lower)
            if skill_match:
                extracted_info["skill"] = skill_match
            return QueryType.SKILL_QUERY, extracted_info
        
        # Conditional Edge 4: Consultas sobre experiencia
        if self._match_patterns(query_lower, QueryType.EXPERIENCE_QUERY):
            return QueryType.EXPERIENCE_QUERY, extracted_info
        
        # Conditional Edge 5: Consultas sobre educación
        if self._match_patterns(query_lower, QueryType.EDUCATION_QUERY):
            return QueryType.EDUCATION_QUERY, extracted_info
        
        # Conditional Edge 6: Consultas sobre contacto
        if self._match_patterns(query_lower, QueryType.CONTACT_QUERY):
            return QueryType.CONTACT_QUERY, extracted_info
        
        # Conditional Edge 7: Información general
        return QueryType.GENERAL_INFO, extracted_info
    
    def _match_patterns(self, query: str, query_type: QueryType) -> bool:
        """Verifica si la consulta coincide con los patrones de un tipo específico"""
        patterns = self.patterns.get(query_type, [])
        for pattern in patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _extract_student_name(self, query: str) -> str:
        """Extrae el nombre del estudiante de la consulta"""
        patterns = self.patterns[QueryType.STUDENT_SEARCH]
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def _extract_skill(self, query: str) -> str:
        """Extrae habilidades específicas mencionadas en la consulta"""
        # Patrones para extraer habilidades específicas
        skill_patterns = [
            r"(?:sabe|conoce|domina)\s+(?:de\s+)?(\w+)",
            r"experiencia en\s+(\w+)",
            r"conocimientos?\s+(?:de|en)\s+(\w+)"
        ]
        
        for pattern in skill_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None
    
    def get_search_strategy(self, query_type: QueryType, extracted_info: Dict) -> Dict:
        """
        Determina la estrategia de búsqueda basada en el tipo de consulta
        """
        strategy = {
            "use_vector_search": True,
            "search_filters": {},
            "response_template": "default",
            "max_results": 5
        }
        
        if query_type == QueryType.STUDENT_SEARCH:
            strategy["search_filters"]["student_name"] = extracted_info.get("student_name")
            strategy["response_template"] = "student_profile"
            strategy["max_results"] = 1
        
        elif query_type == QueryType.SKILL_QUERY:
            strategy["search_filters"]["skill"] = extracted_info.get("skill")
            strategy["response_template"] = "skill_focused"
            strategy["max_results"] = 3
        
        elif query_type == QueryType.EXPERIENCE_QUERY:
            strategy["response_template"] = "experience_focused"
            strategy["max_results"] = 5
        
        elif query_type == QueryType.EDUCATION_QUERY:
            strategy["response_template"] = "education_focused"
            strategy["max_results"] = 5
        
        elif query_type == QueryType.CONTACT_QUERY:
            strategy["response_template"] = "contact_focused"
            strategy["max_results"] = 3
        
        elif query_type == QueryType.GREETING:
            strategy["use_vector_search"] = False
            strategy["response_template"] = "greeting"
        
        return strategy
    
    def should_use_rag(self, query_type: QueryType) -> bool:
        """
        Conditional edge para determinar si se debe usar RAG
        """
        non_rag_types = {QueryType.GREETING, QueryType.UNKNOWN}
        return query_type not in non_rag_types

# Clase para gestionar las respuestas basadas en templates
class ResponseTemplates:
    """Plantillas de respuesta basadas en el tipo de consulta"""
    
    @staticmethod
    def get_template(template_type: str) -> str:
        templates = {
            "greeting": """¡Hola! Soy tu asistente para consultas sobre CVs de estudiantes. 
            Puedo ayudarte a:
            - Buscar información específica de estudiantes
            - Consultar habilidades y competencias
            - Revisar experiencia laboral
            - Verificar información educativa
            - Obtener datos de contacto
            
            ¿En qué puedo ayudarte hoy?""",
            
            "student_profile": """Basándome en la información encontrada sobre {student_name}:
            
            {context}
            
            Esta información proviene de los CVs almacenados en nuestra base de datos.""",
            
            "skill_focused": """Información sobre habilidades en {skill}:
            
            {context}
            
            Estos datos han sido extraídos de los CVs de estudiantes disponibles.""",
            
            "experience_focused": """Información sobre experiencia laboral:
            
            {context}
            
            Datos recopilados de los CVs de estudiantes.""",
            
            "education_focused": """Información educativa encontrada:
            
            {context}
            
            Información extraída de los CVs académicos.""",
            
            "contact_focused": """Datos de contacto disponibles:
            
            {context}
            
            Información de contacto de los CVs.""",
            
            "default": """Basándome en la información disponible:
            
            {context}
            
            Esta respuesta se basa en los CVs de estudiantes almacenados."""
        }
        
        return templates.get(template_type, templates["default"])