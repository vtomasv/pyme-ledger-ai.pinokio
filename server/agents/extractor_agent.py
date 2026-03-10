"""
Agente Extractor para extracción de campos clave de documentos.
Usa LLM local (Ollama) para identificar: proveedor, RUT, fecha, folio, montos, etc.
"""
import os
import json
import re
from typing import Dict, Optional
from datetime import datetime
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TIMEOUT_EXTRACTION = int(os.getenv("OLLAMA_TIMEOUT_EXTRACTION", "300"))


class ExtractorAgent:
    """Agente especializado en extracción de campos clave de documentos contables."""
    
    def __init__(self, model: str = "llama3.2:3b"):
        """
        Inicializa el agente extractor.
        
        Args:
            model: Modelo Ollama a usar
        """
        self.model = model
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Retorna el system prompt para extracción."""
        return """Eres un experto contador y procesador de documentos contables.
Tu tarea es extraer información clave de documentos contables (facturas, boletas, cartolas).

IMPORTANTE: Responde SIEMPRE en formato JSON válido, sin explicaciones adicionales.

Campos a extraer:
- proveedor: Nombre de la empresa/persona que emite el documento
- rut_proveedor: RUT o ID del proveedor (formato: XX.XXX.XXX-X)
- fecha_emision: Fecha del documento (formato: YYYY-MM-DD)
- folio: Número de folio/factura
- monto_neto: Monto sin IVA (número sin separadores)
- iva: Monto del IVA (número sin separadores)
- monto_total: Monto total (número sin separadores)
- moneda: Código de moneda (CLP, USD, EUR, etc.)
- tipo_documento: Tipo (Factura, Boleta, Cartola, Otro)
- observaciones: Notas relevantes

Si no encuentras un campo, usa null. Normaliza montos a números sin puntos ni comas.
Responde SOLO con JSON, sin markdown, sin explicaciones."""
    
    def extract_fields(self, text: str) -> Dict[str, any]:
        """
        Extrae campos clave del texto de un documento.
        
        Args:
            text: Texto extraído del documento (OCR)
            
        Returns:
            Dict con campos extraídos
        """
        if not text or len(text.strip()) < 50:
            return self._get_empty_fields()
        
        # Limitar texto a primeros 4000 caracteres para no saturar el contexto
        text_limited = text[:4000]
        
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"Extrae los campos de este documento:\n\n{text_limited}"}
                    ],
                    "options": {"temperature": 0.2},  # Bajo para precisión
                    "stream": False
                },
                timeout=TIMEOUT_EXTRACTION
            )
            response.encoding = "utf-8"
            response.raise_for_status()
            
            content = response.json()["message"]["content"]
            extracted = self._parse_extraction_response(content)
            
            # Validar y normalizar
            extracted = self._validate_fields(extracted)
            
            return extracted
        
        except Exception as e:
            print(f"Error en extracción: {e}")
            return self._get_empty_fields()
    
    def _parse_extraction_response(self, response: str) -> Dict:
        """Parsea la respuesta JSON del LLM."""
        try:
            # Limpiar markdown si existe
            clean = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`").strip()
            
            # Intentar parsear JSON
            return json.loads(clean)
        except json.JSONDecodeError:
            # Fallback: buscar JSON entre llaves
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    return json.loads(response[start:end])
                except:
                    pass
            
            return self._get_empty_fields()
    
    def _validate_fields(self, fields: Dict) -> Dict:
        """Valida y normaliza los campos extraídos."""
        validated = {
            "proveedor": self._clean_string(fields.get("proveedor")),
            "rut_proveedor": self._clean_rut(fields.get("rut_proveedor")),
            "fecha_emision": self._parse_date(fields.get("fecha_emision")),
            "folio": self._clean_string(fields.get("folio")),
            "monto_neto": self._parse_amount(fields.get("monto_neto")),
            "iva": self._parse_amount(fields.get("iva")),
            "monto_total": self._parse_amount(fields.get("monto_total")),
            "moneda": self._clean_currency(fields.get("moneda", "CLP")),
            "tipo_documento": self._clean_string(fields.get("tipo_documento")),
            "observaciones": self._clean_string(fields.get("observaciones"))
        }
        
        # Validación cruzada: neto + IVA ≈ total
        if validated["monto_neto"] and validated["iva"] and validated["monto_total"]:
            calculated = validated["monto_neto"] + validated["iva"]
            if abs(calculated - validated["monto_total"]) > 1:  # Tolerancia de 1
                validated["_validation_warning"] = "Inconsistencia: neto + IVA ≠ total"
        
        return validated
    
    def _clean_string(self, value: any) -> Optional[str]:
        """Limpia y valida strings."""
        if value is None:
            return None
        s = str(value).strip()
        return s if s else None
    
    def _clean_rut(self, value: any) -> Optional[str]:
        """Limpia y valida RUT chileno."""
        if value is None:
            return None
        # Remover caracteres no alfanuméricos excepto guión
        rut = re.sub(r"[^\d\-K]", "", str(value).upper())
        # Validar formato XX.XXX.XXX-X o XXXXXXXX-X
        if re.match(r"^\d{1,2}\.\d{3}\.\d{3}-[\dK]$", rut) or re.match(r"^\d{7,8}-[\dK]$", rut):
            return rut
        return None
    
    def _parse_date(self, value: any) -> Optional[str]:
        """Parsea y valida fechas."""
        if value is None:
            return None
        
        date_str = str(value).strip()
        
        # Formatos comunes
        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d-%m-%Y",
            "%d/%m/%y",
            "%Y/%m/%d"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        return None
    
    def _parse_amount(self, value: any) -> Optional[float]:
        """Parsea y valida montos."""
        if value is None:
            return None
        
        try:
            # Remover caracteres no numéricos excepto punto y coma
            s = str(value).strip()
            # Reemplazar comas por puntos si es necesario
            s = s.replace(",", ".")
            # Remover otros caracteres
            s = re.sub(r"[^\d\.]", "", s)
            
            amount = float(s)
            return amount if amount >= 0 else None
        except (ValueError, AttributeError):
            return None
    
    def _clean_currency(self, value: any) -> str:
        """Limpia y valida código de moneda."""
        if value is None:
            return "CLP"
        
        code = str(value).strip().upper()
        # Validar que sea código de 3 letras
        if re.match(r"^[A-Z]{3}$", code):
            return code
        
        return "CLP"  # Default
    
    def _get_empty_fields(self) -> Dict:
        """Retorna estructura vacía de campos."""
        return {
            "proveedor": None,
            "rut_proveedor": None,
            "fecha_emision": None,
            "folio": None,
            "monto_neto": None,
            "iva": None,
            "monto_total": None,
            "moneda": "CLP",
            "tipo_documento": None,
            "observaciones": None,
            "_extraction_error": "No se pudo extraer información"
        }
    
    def batch_extract(self, documents: list) -> list:
        """
        Extrae campos de múltiples documentos.
        
        Args:
            documents: Lista de textos de documentos
            
        Returns:
            Lista de campos extraídos
        """
        results = []
        for i, text in enumerate(documents):
            result = self.extract_fields(text)
            result["_index"] = i
            results.append(result)
        
        return results


# Instancia global
extractor_agent = None


def get_extractor_agent(model: str = "llama3.2:3b") -> ExtractorAgent:
    """Obtiene o crea la instancia del agente extractor."""
    global extractor_agent
    if extractor_agent is None:
        extractor_agent = ExtractorAgent(model)
    return extractor_agent
