"""
Agente OCR para extracción de texto de documentos.
Soporta Tesseract, EasyOCR y extracción directa de PDFs digitales.
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Tuple
import hashlib
from PIL import Image
import PyPDF2
import pytesseract
import easyocr

# Configuración
TESSERACT_PATH = os.environ.get("TESSERACT_PATH", "tesseract")
OCR_LANG = "spa+eng"  # Español + Inglés


class OCRAgent:
    """Agente especializado en extracción de texto de documentos."""
    
    def __init__(self, hardware_profile: str = "cpu"):
        """
        Inicializa el agente OCR.
        
        Args:
            hardware_profile: "cpu", "gpu_light", "gpu_heavy"
        """
        self.hardware_profile = hardware_profile
        self.reader = None
        self._init_easyocr()
    
    def _init_easyocr(self):
        """Inicializa EasyOCR si está disponible."""
        try:
            gpu = self.hardware_profile.startswith("gpu")
            self.reader = easyocr.Reader(["es", "en"], gpu=gpu)
        except Exception as e:
            print(f"⚠️ EasyOCR no disponible: {e}. Usando Tesseract como fallback.")
            self.reader = None
    
    def extract_from_file(self, file_path: str) -> Dict[str, any]:
        """
        Extrae texto de un archivo (PDF, JPG, PNG).
        
        Args:
            file_path: Ruta al archivo
            
        Returns:
            Dict con texto extraído y metadata
        """
        path = Path(file_path)
        
        if not path.exists():
            return {"error": f"Archivo no encontrado: {file_path}", "text": ""}
        
        file_ext = path.suffix.lower()
        
        try:
            if file_ext == ".pdf":
                return self._extract_from_pdf(file_path)
            elif file_ext in [".jpg", ".jpeg", ".png"]:
                return self._extract_from_image(file_path)
            else:
                return {"error": f"Formato no soportado: {file_ext}", "text": ""}
        except Exception as e:
            return {"error": f"Error extrayendo texto: {str(e)}", "text": ""}
    
    def _extract_from_pdf(self, file_path: str) -> Dict[str, any]:
        """Extrae texto de un PDF."""
        path = Path(file_path)
        text_parts = []
        pages_processed = 0
        
        try:
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                
                for page_num, page in enumerate(reader.pages):
                    # Intentar extracción directa primero (PDF digital)
                    text = page.extract_text()
                    
                    if text and len(text.strip()) > 50:
                        # PDF digital con texto extraíble
                        text_parts.append(text)
                        pages_processed += 1
                    else:
                        # PDF escaneado, usar OCR
                        try:
                            # Convertir página a imagen y aplicar OCR
                            images = self._pdf_page_to_images(file_path, page_num)
                            for img in images:
                                ocr_text = self._ocr_image(img)
                                if ocr_text:
                                    text_parts.append(ocr_text)
                            pages_processed += 1
                        except Exception as e:
                            print(f"Error procesando página {page_num}: {e}")
                
                full_text = "\n---PÁGINA---\n".join(text_parts)
                
                return {
                    "text": full_text,
                    "pages": num_pages,
                    "pages_processed": pages_processed,
                    "method": "pdf_mixed",
                    "hash": self._compute_hash(file_path)
                }
        
        except Exception as e:
            return {"error": f"Error procesando PDF: {str(e)}", "text": ""}
    
    def _extract_from_image(self, file_path: str) -> Dict[str, any]:
        """Extrae texto de una imagen."""
        try:
            img = Image.open(file_path)
            text = self._ocr_image(img)
            
            return {
                "text": text,
                "method": "image_ocr",
                "size": img.size,
                "hash": self._compute_hash(file_path)
            }
        except Exception as e:
            return {"error": f"Error procesando imagen: {str(e)}", "text": ""}
    
    def _ocr_image(self, image: Image.Image) -> str:
        """Aplica OCR a una imagen usando EasyOCR o Tesseract."""
        if self.reader:
            # Usar EasyOCR (más preciso)
            try:
                results = self.reader.readtext(image, detail=0)
                return "\n".join(results)
            except Exception as e:
                print(f"Error con EasyOCR: {e}")
        
        # Fallback a Tesseract
        try:
            text = pytesseract.image_to_string(image, lang=OCR_LANG)
            return text
        except Exception as e:
            print(f"Error con Tesseract: {e}")
            return ""
    
    def _pdf_page_to_images(self, pdf_path: str, page_num: int) -> list:
        """Convierte una página PDF a imagen(s)."""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, first_page=page_num+1, last_page=page_num+1)
            return images
        except Exception as e:
            print(f"Error convirtiendo PDF a imagen: {e}")
            return []
    
    def _compute_hash(self, file_path: str) -> str:
        """Calcula hash SHA256 del archivo."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def batch_extract(self, folder_path: str, file_extensions: list = None) -> list:
        """
        Extrae texto de múltiples archivos en una carpeta.
        
        Args:
            folder_path: Ruta a la carpeta
            file_extensions: Extensiones a procesar (default: [".pdf", ".jpg", ".png"])
            
        Returns:
            Lista de resultados
        """
        if file_extensions is None:
            file_extensions = [".pdf", ".jpg", ".jpeg", ".png"]
        
        folder = Path(folder_path)
        results = []
        
        for file_path in folder.rglob("*"):
            if file_path.suffix.lower() in file_extensions:
                result = self.extract_from_file(str(file_path))
                result["file"] = str(file_path)
                results.append(result)
        
        return results


# Instancia global
ocr_agent = None


def get_ocr_agent(hardware_profile: str = "cpu") -> OCRAgent:
    """Obtiene o crea la instancia del agente OCR."""
    global ocr_agent
    if ocr_agent is None:
        ocr_agent = OCRAgent(hardware_profile)
    return ocr_agent
