"""
Agente Clasificador para clasificación automática de gastos.
Usa reglas, keywords y LLM local para sugerir categoría y centro de costo.
"""
import os
import json
import re
from typing import Dict, Optional, Tuple
import requests
from difflib import SequenceMatcher

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TIMEOUT_CLASSIFICATION = int(os.getenv("OLLAMA_TIMEOUT_CLASSIFICATION", "300"))


class ClassifierAgent:
    """Agente especializado en clasificación de gastos contables."""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa el agente clasificador.
        
        Args:
            model: Modelo Ollama a usar
        """
        self.model = model
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Retorna el system prompt para clasificación."""
        return """Eres un experto contador especializado en clasificación de gastos.
Tu tarea es clasificar gastos en categorías contables basándote en:
- Nombre del proveedor
- Descripción del gasto
- Monto
- Contexto empresarial

Responde SIEMPRE en formato JSON válido:
{
  "categoria_sugerida": "Nombre de la categoría",
  "confianza": 0.95,
  "razon": "Breve explicación",
  "centro_costo_sugerido": "Código o nombre del centro",
  "excepciones": ["Lista de alertas si las hay"]
}

Confianza: 0.0 a 1.0 (1.0 = muy seguro, 0.5 = dudoso, 0.0 = no clasificable)
Si no puedes clasificar, usa confianza baja (0.3-0.5) y explica en excepciones.
Responde SOLO con JSON, sin markdown."""
    
    def classify(self, document_data: Dict, categories: list, cost_centers: list) -> Dict:
        """
        Clasifica un documento en categoría y centro de costo.
        
        Args:
            document_data: Dict con campos extraídos (proveedor, monto, etc.)
            categories: Lista de categorías contables disponibles
            cost_centers: Lista de centros de costo disponibles
            
        Returns:
            Dict con clasificación sugerida
        """
        # Paso 1: Intentar clasificación por reglas (rápida)
        rule_result = self._classify_by_rules(document_data, categories)
        if rule_result.get("confianza", 0) > 0.8:
            return rule_result
        
        # Paso 2: Intentar clasificación por keywords
        keyword_result = self._classify_by_keywords(document_data, categories)
        if keyword_result.get("confianza", 0) > 0.7:
            return keyword_result
        
        # Paso 3: Usar LLM para clasificación semántica
        llm_result = self._classify_by_llm(document_data, categories, cost_centers)
        
        # Paso 4: Sugerir centro de costo
        if not llm_result.get("centro_costo_sugerido"):
            llm_result["centro_costo_sugerido"] = self._suggest_cost_center(
                llm_result.get("categoria_sugerida"),
                cost_centers
            )
        
        return llm_result
    
    def _classify_by_rules(self, document_data: Dict, categories: list) -> Dict:
        """Clasificación por reglas duras (proveedores conocidos, montos, etc.)."""
        proveedor = (document_data.get("proveedor") or "").lower()
        monto = document_data.get("monto_total") or 0
        
        # Reglas básicas por proveedor
        provider_rules = {
            "serv": "Servicios",
            "agua": "Servicios",
            "electricidad": "Servicios",
            "teléfono": "Servicios",
            "internet": "Servicios",
            "arriendo": "Arriendo",
            "rent": "Arriendo",
            "google": "Marketing",
            "facebook": "Marketing",
            "instagram": "Marketing",
            "publicidad": "Marketing",
            "imprenta": "Marketing",
            "banco": "Financiero",
            "impuesto": "Impuestos",
            "sii": "Impuestos",
            "afp": "Impuestos",
            "isapre": "Impuestos",
        }
        
        for keyword, category in provider_rules.items():
            if keyword in proveedor:
                return {
                    "categoria_sugerida": category,
                    "confianza": 0.85,
                    "razon": f"Clasificación por proveedor: {proveedor}",
                    "centro_costo_sugerido": None,
                    "excepciones": []
                }
        
        return {"confianza": 0}
    
    def _classify_by_keywords(self, document_data: Dict, categories: list) -> Dict:
        """Clasificación por keywords en descripción."""
        proveedor = (document_data.get("proveedor") or "").lower()
        observaciones = (document_data.get("observaciones") or "").lower()
        
        text_to_search = f"{proveedor} {observaciones}"
        
        best_match = None
        best_score = 0
        
        for category in categories:
            keywords = category.get("keywords", [])
            if isinstance(keywords, str):
                try:
                    keywords = json.loads(keywords)
                except:
                    keywords = []
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in text_to_search:
                    score = len(keyword) / len(text_to_search)  # Normalizar por longitud
                    if score > best_score:
                        best_score = score
                        best_match = category
        
        if best_match and best_score > 0.3:
            return {
                "categoria_sugerida": best_match.get("nombre"),
                "confianza": min(0.75, best_score),
                "razon": f"Coincidencia de keywords",
                "centro_costo_sugerido": None,
                "excepciones": []
            }
        
        return {"confianza": 0}
    
    def _classify_by_llm(self, document_data: Dict, categories: list, cost_centers: list) -> Dict:
        """Clasificación usando LLM local."""
        proveedor = document_data.get("proveedor") or "Desconocido"
        monto = document_data.get("monto_total") or 0
        observaciones = document_data.get("observaciones") or ""
        
        # Preparar contexto con categorías disponibles
        categories_text = "\n".join([
            f"- {c.get('nombre')} ({c.get('tipo_gasto')})"
            for c in categories[:10]  # Limitar a 10 para no saturar
        ])
        
        cost_centers_text = "\n".join([
            f"- {cc.get('codigo')}: {cc.get('nombre')}"
            for cc in cost_centers[:10]
        ])
        
        prompt = f"""Clasifica este gasto:
Proveedor: {proveedor}
Monto: {monto}
Observaciones: {observaciones}

Categorías disponibles:
{categories_text}

Centros de costo disponibles:
{cost_centers_text}"""
        
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "options": {"temperature": 0.3},
                    "stream": False
                },
                timeout=TIMEOUT_CLASSIFICATION
            )
            response.encoding = "utf-8"
            response.raise_for_status()
            
            content = response.json()["message"]["content"]
            result = self._parse_classification_response(content)
            
            return result
        
        except Exception as e:
            print(f"Error en clasificación LLM: {e}")
            return {
                "categoria_sugerida": None,
                "confianza": 0.0,
                "razon": f"Error en clasificación: {str(e)}",
                "centro_costo_sugerido": None,
                "excepciones": ["Error al consultar LLM"]
            }
    
    def _parse_classification_response(self, response: str) -> Dict:
        """Parsea la respuesta JSON del LLM."""
        try:
            # Limpiar markdown
            clean = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()
            result = json.loads(clean)
            
            # Validar campos
            result.setdefault("categoria_sugerida", None)
            result.setdefault("confianza", 0.5)
            result.setdefault("razon", "")
            result.setdefault("centro_costo_sugerido", None)
            result.setdefault("excepciones", [])
            
            # Asegurar que confianza está entre 0 y 1
            result["confianza"] = max(0.0, min(1.0, float(result.get("confianza", 0.5))))
            
            return result
        
        except Exception as e:
            print(f"Error parseando clasificación: {e}")
            return {
                "categoria_sugerida": None,
                "confianza": 0.0,
                "razon": "Error al parsear respuesta",
                "centro_costo_sugerido": None,
                "excepciones": ["Error de parseo"]
            }
    
    def _suggest_cost_center(self, category_name: Optional[str], cost_centers: list) -> Optional[str]:
        """Sugiere un centro de costo basado en la categoría."""
        if not category_name or not cost_centers:
            return None
        
        # Buscar coincidencia por nombre
        for cc in cost_centers:
            if category_name.lower() in cc.get("nombre", "").lower():
                return cc.get("codigo")
        
        # Retornar el primero como default
        return cost_centers[0].get("codigo") if cost_centers else None
    
    def batch_classify(self, documents: list, categories: list, cost_centers: list) -> list:
        """
        Clasifica múltiples documentos.
        
        Args:
            documents: Lista de datos de documentos
            categories: Lista de categorías
            cost_centers: Lista de centros de costo
            
        Returns:
            Lista de clasificaciones
        """
        results = []
        for doc in documents:
            result = self.classify(doc, categories, cost_centers)
            result["_document_id"] = doc.get("_id")
            results.append(result)
        
        return results


# Instancia global
classifier_agent = None


def get_classifier_agent(model: str = "llama3.2:3b") -> ClassifierAgent:
    """Obtiene o crea la instancia del agente clasificador."""
    global classifier_agent
    if classifier_agent is None:
        classifier_agent = ClassifierAgent(model)
    return classifier_agent
