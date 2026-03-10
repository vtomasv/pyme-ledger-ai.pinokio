"""
Agente OCR para extracción de texto de documentos.
Estrategia de fallbacks robusta:
  1. PDF digital → PyPDF2 extracción directa (sin dependencias externas)
  2. PDF escaneado → pdf2image + OCR
  3. Imagen → EasyOCR (lazy load) → Tesseract → vacío sin crashear
"""
import hashlib
import os
from pathlib import Path
from typing import Dict, Optional

# ── Imports opcionales ────────────────────────────────────────────────────────
try:
    import PyPDF2
    _PYPDF2_AVAILABLE = True
except ImportError:
    _PYPDF2_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

# EasyOCR: lazy load para no bloquear el arranque del servidor
_easyocr_reader = None
_easyocr_tried = False


def _get_easyocr_reader():
    """Carga EasyOCR una sola vez de forma lazy."""
    global _easyocr_reader, _easyocr_tried
    if _easyocr_tried:
        return _easyocr_reader
    _easyocr_tried = True
    try:
        import easyocr
        _easyocr_reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
        print("INFO: EasyOCR cargado correctamente")
    except Exception as e:
        print(f"INFO: EasyOCR no disponible ({e})")
        _easyocr_reader = None
    return _easyocr_reader


class OCRAgent:
    """Agente OCR con múltiples estrategias de extracción."""

    def __init__(self, hardware_profile: str = "cpu"):
        self.hardware_profile = hardware_profile
        print(f"INFO: OCRAgent inicializado (perfil: {hardware_profile})")

    def extract_from_file(self, file_path: str) -> Dict:
        """
        Extrae texto de un archivo. Nunca lanza excepciones.
        Siempre retorna dict con clave 'text'.
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        if not path.exists():
            return {"text": "", "method": "error",
                    "error": f"Archivo no encontrado: {file_path}"}

        try:
            file_hash = self._compute_hash(file_path)
        except Exception:
            file_hash = ""

        if ext == ".pdf":
            result = self._extract_pdf(file_path)
        elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"):
            result = self._extract_image(file_path)
        else:
            result = {"text": "", "method": "unsupported",
                      "error": f"Extensión no soportada: {ext}"}

        result["hash"] = file_hash
        result["file"] = str(file_path)
        if "text" not in result:
            result["text"] = ""

        chars = len(result.get("text", ""))
        print(f"INFO: OCR completado — método: {result.get('method','?')}, chars: {chars}")
        return result

    # ── PDF ──────────────────────────────────────────────────────────────────

    def _extract_pdf(self, file_path: str) -> Dict:
        """Extrae texto de PDF: digital primero, OCR si está escaneado."""
        if not _PYPDF2_AVAILABLE:
            return {"text": "", "method": "error", "error": "PyPDF2 no disponible"}

        try:
            text_parts = []
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)

                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""

                    if len(page_text.strip()) >= 20:
                        text_parts.append(page_text)
                    else:
                        # PDF escaneado: intentar OCR vía pdf2image
                        ocr_text = self._pdf_page_to_ocr(file_path, page_num)
                        if ocr_text:
                            text_parts.append(ocr_text)
                        elif page_text:
                            text_parts.append(page_text)

            full_text = "\n\n".join(text_parts).strip()
            return {
                "text": full_text,
                "method": "pdf_text" if full_text else "pdf_empty",
                "pages": num_pages
            }
        except Exception as e:
            return {"text": "", "method": "pdf_error", "error": str(e)}

    def _pdf_page_to_ocr(self, pdf_path: str, page_num: int) -> str:
        """Convierte página PDF a imagen y aplica OCR."""
        try:
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path,
                                       first_page=page_num + 1,
                                       last_page=page_num + 1,
                                       dpi=200)
            if images:
                return self._ocr_pil_image(images[0])
        except Exception as e:
            print(f"INFO: pdf2image no disponible ({e})")
        return ""

    # ── IMAGEN ───────────────────────────────────────────────────────────────

    def _extract_image(self, file_path: str) -> Dict:
        """Extrae texto de una imagen con múltiples fallbacks."""
        if not _PIL_AVAILABLE:
            return {"text": "", "method": "error", "error": "Pillow no disponible"}

        try:
            img = Image.open(file_path)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            text = self._ocr_pil_image(img)
            return {"text": text, "method": "image_ocr", "size": list(img.size)}
        except Exception as e:
            return {"text": "", "method": "image_error", "error": str(e)}

    def _ocr_pil_image(self, img) -> str:
        """
        OCR sobre imagen PIL con fallbacks:
        1. EasyOCR (lazy load, más preciso)
        2. Tesseract (si está instalado en el sistema)
        3. Retorna vacío sin crashear
        """
        # Intento 1: EasyOCR
        try:
            reader = _get_easyocr_reader()
            if reader is not None:
                import numpy as np
                img_array = np.array(img)
                results = reader.readtext(img_array, detail=0, paragraph=True)
                text = "\n".join(str(r) for r in results)
                if text.strip():
                    return text
        except Exception as e:
            print(f"INFO: EasyOCR readtext falló ({e})")

        # Intento 2: Tesseract
        if _TESSERACT_AVAILABLE:
            try:
                text = pytesseract.image_to_string(img, lang="spa+eng", config="--psm 3")
                if text.strip():
                    return text
            except Exception as e:
                print(f"INFO: Tesseract falló ({e})")

        return ""

    # ── UTILIDADES ───────────────────────────────────────────────────────────

    def _compute_hash(self, file_path: str) -> str:
        """SHA256 del archivo para detección de duplicados."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


# ── Singleton ─────────────────────────────────────────────────────────────────
_ocr_agent_instance: Optional[OCRAgent] = None


def get_ocr_agent(hardware_profile: str = "cpu") -> OCRAgent:
    """Obtiene o crea la instancia singleton del agente OCR."""
    global _ocr_agent_instance
    if _ocr_agent_instance is None:
        _ocr_agent_instance = OCRAgent(hardware_profile)
    return _ocr_agent_instance
