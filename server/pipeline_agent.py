"""
pipeline_agent.py — Pipeline Agéntico de Procesamiento de Documentos
======================================================================
Arquitectura de 6 pasos con streaming SSE:
  Paso 1: OCR         — Tesseract multi-PSM + preprocesamiento de imagen
  Paso 2: Visión VLLM — moondream/llava/qwen2.5-vl si disponible en Ollama
  Paso 3: Extracción  — Regex robusto + LLM (qwen2.5/llama3.2) para campos
  Paso 4: Clasificación — Keywords semánticos + LLM para categoría contable
  Paso 5: Auditoría   — Validación, duplicados, anomalías
  Paso 6: Guardado    — Persiste en SQLite con archivo en disco

Streaming via SSE: cada paso emite eventos JSON al cliente en tiempo real.
Compatible con macOS, Windows y Linux.
"""

import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# ── Modelos preferidos (en orden de prioridad) ────────────────────────────────
# Modelo fijo: qwen3.5:0.8b — único modelo soportado
QWEN_MODEL = "qwen3.5:0.8b"

# Modelo ÚNICO — no se usa ningún otro
TEXT_MODELS = ["qwen3.5:0.8b"]
VISION_MODELS = ["qwen3.5:0.8b"]  # qwen3.5 también puede analizar imágenes con base64

# ── Utilidades Ollama ─────────────────────────────────────────────────────────

def _ollama_list() -> List[str]:
    """Lista modelos disponibles en Ollama."""
    try:
        import requests
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def _find_best_model(preferred_list: List[str], available: List[str]) -> Optional[str]:
    """Encuentra el mejor modelo disponible de la lista de preferidos."""
    available_lower = {m.lower(): m for m in available}
    available_bases: Dict[str, str] = {}
    for m in available:
        base = m.split(":")[0].lower()
        if base not in available_bases:
            available_bases[base] = m

    for preferred in preferred_list:
        p_lower = preferred.lower()
        p_base = p_lower.split(":")[0]
        if p_lower in available_lower:
            return available_lower[p_lower]
        if p_base in available_bases:
            return available_bases[p_base]
    return None


def _ollama_generate(model: str, prompt: str, image_b64: Optional[str] = None,
                     timeout: int = 120) -> str:
    """Llama a Ollama y retorna la respuesta completa."""
    try:
        import requests
        payload: Dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1000, "top_p": 0.9}
        }
        if image_b64:
            payload["images"] = [image_b64]
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=timeout)
        if r.status_code == 200:
            return r.json().get("response", "").strip()
        logger.warning(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.debug(f"Ollama error: {e}")
    return ""


# ── OCR con Tesseract ─────────────────────────────────────────────────────────

def _preprocess_image(img_path: str) -> List[str]:
    """Genera variantes preprocesadas de la imagen para mejorar OCR."""
    variants = [img_path]
    try:
        from PIL import Image, ImageFilter, ImageEnhance
        img = Image.open(img_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size
        scale = max(1.5, 1200 / max(w, 1))
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        gray = img.convert("L")

        # Variante 1: contraste alto + sharpen doble + threshold 180
        # Ideal para boletas con fondo cuadriculado o ruidoso
        v1 = ImageEnhance.Contrast(gray).enhance(3.0)
        v1 = v1.filter(ImageFilter.SHARPEN)
        v1 = v1.filter(ImageFilter.SHARPEN)
        v1 = v1.point(lambda x: 0 if x < 180 else 255, "1").convert("L")
        tmp1 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        v1.save(tmp1.name, "PNG", dpi=(300, 300))
        tmp1.close()
        variants.append(tmp1.name)

        # Variante 2: contraste moderado sin binarización
        # Ideal para documentos con texto de color o gris
        v2 = ImageEnhance.Contrast(gray).enhance(2.0)
        v2 = v2.filter(ImageFilter.SHARPEN)
        tmp2 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        v2.save(tmp2.name, "PNG", dpi=(300, 300))
        tmp2.close()
        variants.append(tmp2.name)

        # Variante 3: binarización suave (threshold 140)
        # Ideal para documentos muy oscuros o de bajo contraste
        v3 = gray.point(lambda x: 0 if x < 140 else 255, "1").convert("L")
        tmp3 = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        v3.save(tmp3.name, "PNG", dpi=(300, 300))
        tmp3.close()
        variants.append(tmp3.name)

    except Exception as e:
        logger.debug(f"Preprocesamiento: {e}")
    return variants


def _run_tesseract(img_path: str) -> str:
    """Corre Tesseract con múltiples PSM y retorna el mejor resultado."""
    best, best_words = "", 0
    for psm in [6, 11, 4, 3]:
        try:
            r = subprocess.run(
                ["tesseract", img_path, "stdout", "-l", "spa+eng",
                 "--psm", str(psm), "--oem", "3"],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode == 0:
                text = r.stdout.strip()
                words = len([w for w in text.split() if len(w) > 2])
                if words > best_words:
                    best_words = words
                    best = text
                if best_words > 60:
                    break
        except Exception:
            pass
    return best


def _ocr_quality_score(text: str) -> float:
    """
    Calcula un puntaje de calidad del texto OCR.
    Penaliza caracteres de ruido (\\, /, |, S sueltas) y premia palabras reales.
    """
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    # Contar palabras reales (>2 chars, mayormente alfanuméricas)
    real_words = [w for w in words if len(w) > 2 and re.match(r'[A-Za-z0-9áéíóúÁÉÍÓÚñÑ]', w)]
    # Penalizar tokens de ruido (letras sueltas, símbolos)
    noise_tokens = [w for w in words if re.match(r'^[\\|/\-_=~<>\^]+$', w) or (len(w) == 1 and not w.isdigit())]
    noise_ratio = len(noise_tokens) / max(len(words), 1)
    score = len(real_words) * (1.0 - noise_ratio)
    return score


def ocr_image(img_path: str) -> Dict:
    """OCR completo sobre una imagen. Retorna {text, words, method}."""
    variants = _preprocess_image(img_path)
    all_texts = []
    best_text, best_score = "", 0.0
    best_words = 0
    method = "none"

    for v in variants:
        text = _run_tesseract(v)
        if text.strip():
            all_texts.append(text)
            score = _ocr_quality_score(text)
            words = len([w for w in text.split() if len(w) > 2])
            if score > best_score:
                best_score = score
                best_words = words
                best_text = text
                method = "tesseract"

    # Combinar textos de todas las variantes para maximizar cobertura
    # La extraccion de campos se hará sobre el texto combinado
    if len(all_texts) > 1:
        combined = "\n\n".join(all_texts)
        combined_score = _ocr_quality_score(combined)
        # Si el combinado tiene mejor score, usarlo
        if combined_score > best_score * 1.2:
            best_text = combined
            best_words = len([w for w in combined.split() if len(w) > 2])

    # Limpiar temporales
    for v in variants[1:]:
        try:
            os.unlink(v)
        except Exception:
            pass

    # EasyOCR si Tesseract dio poco
    if best_words < 15:
        try:
            import easyocr
            import numpy as np
            from PIL import Image
            reader = easyocr.Reader(["es", "en"], gpu=False, verbose=False)
            img = Image.open(img_path)
            results = reader.readtext(np.array(img), detail=0, paragraph=True)
            easy_text = "\n".join(str(r) for r in results)
            easy_words = len([w for w in easy_text.split() if len(w) > 2])
            if easy_words > best_words:
                best_text = easy_text
                best_words = easy_words
                method = "easyocr"
        except Exception as e:
            logger.debug(f"EasyOCR: {e}")

    return {"text": best_text, "words": best_words, "method": method}


def ocr_pdf(pdf_path: str) -> Dict:
    """OCR para PDFs: texto digital primero, luego imagen."""
    # Texto digital
    try:
        import pypdf
        parts = []
        with open(pdf_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text() or ""
                if t.strip():
                    parts.append(t)
        text = "\n".join(parts)
        words = len([w for w in text.split() if len(w) > 2])
        if words >= 15:
            return {"text": text, "words": words, "method": "pdf_digital"}
    except Exception:
        pass

    # PDF escaneado → imagen → OCR
    try:
        from pdf2image import convert_from_path
        images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=2)
        all_text = []
        for img in images:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img.save(tmp.name, "PNG")
                tmp_name = tmp.name
            result = ocr_image(tmp_name)
            if result["text"]:
                all_text.append(result["text"])
            try:
                os.unlink(tmp_name)
            except Exception:
                pass
        text = "\n".join(all_text)
        return {"text": text, "words": len([w for w in text.split() if len(w) > 2]),
                "method": "pdf_scanned"}
    except Exception as e:
        logger.debug(f"pdf2image: {e}")

    return {"text": "", "words": 0, "method": "failed"}


# ── Extracción de campos por Regex ────────────────────────────────────────────

def _extract_by_regex(text: str) -> Dict:
    """
    Extracción robusta de campos contables por regex.
    Diseñado para facturas/boletas de Chile, Perú, México y Argentina.
    """
    fields: Dict = {}
    if not text:
        return fields

    # ── RUT chileno ──
    rut = re.search(r'R\.?U\.?T\.?\s*[:\s]?\s*(\d{1,2}[.\d]*\d-[\dkK])', text, re.I)
    if rut:
        fields["rut_proveedor"] = rut.group(1).replace(".", "")

    # ── RUC peruano ──
    if "rut_proveedor" not in fields:
        ruc = re.search(r'R\.?U\.?C\.?\s*[:\s]?\s*(\d{8,11})', text, re.I)
        if ruc:
            fields["rut_proveedor"] = ruc.group(1)

    # ── RFC mexicano ──
    if "rut_proveedor" not in fields:
        rfc = re.search(r'RFC\s*[:\s]?\s*([A-Z]{3,4}\d{6}[A-Z0-9]{3})', text, re.I)
        if rfc:
            fields["rut_proveedor"] = rfc.group(1)

    # ── Tipo de documento ──
    tipo_patterns = [
        (r'BOLETA\s+ELECTR[ÓO]NICA', 'BOLETA ELECTRÓNICA'),
        (r'BOLETA\s+DE\s+VENTA', 'BOLETA DE VENTA'),
        (r'BOLETA', 'BOLETA'),
        (r'FACTURA\s+ELECTR[ÓO]NICA', 'FACTURA ELECTRÓNICA'),
        (r'FACTURA', 'FACTURA'),
        (r'NOTA\s+DE\s+CR[ÉE]DITO', 'NOTA DE CRÉDITO'),
        (r'RECIBO\s+DE\s+HONORARIOS', 'RECIBO DE HONORARIOS'),
        (r'RECIBO', 'RECIBO'),
        (r'TICKET', 'TICKET'),
        (r'COMPROBANTE', 'COMPROBANTE'),
    ]
    for pat, tipo in tipo_patterns:
        if re.search(pat, text, re.I):
            fields["tipo_documento"] = tipo
            break

    # ── Folio / Número de documento ──
    folio_patterns = [
        # Número solo después del tipo de documento
        r'(?:BOLETA|FACTURA)\s+(?:ELECTR[ÓO]NICA|DE\s+VENTA)\s*[\n\r]+\s*(\d{1,8})\s*[\n\r]',
        # N° explícito
        r'N[°o]\s*[:\s]?\s*(\d{4,8})',
        r'N[°o]\s+(\d{3,8})',
        r'001[-\s]+N[°o]?\s+(\d{4,8})',
        r'FOLIO\s*[:\s]?\s*(\d{3,8})',
        r'SERIE\s+\w+\s+N[°o]?\s*(\d{4,8})',
    ]
    for pat in folio_patterns:
        m = re.search(pat, text, re.I | re.MULTILINE)
        if m:
            fields["folio"] = m.group(1)
            break

    # ── Fecha ──
    date_patterns = [
        r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})',
        r'(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})',
    ]
    for pat in date_patterns:
        m = re.search(pat, text)
        if m:
            d, mo, y = m.group(1), m.group(2), m.group(3)
            if len(y) == 2:
                y = "20" + y
            try:
                # Validar que sea una fecha razonable
                day_int = int(d)
                mon_int = int(mo)
                if 1 <= day_int <= 31 and 1 <= mon_int <= 12:
                    fields["fecha_emision"] = f"{y}-{mo.zfill(2)}-{d.zfill(2)}"
                    break
            except Exception:
                pass

    # ── Montos — requieren mínimo 3 dígitos para evitar falsos positivos ──
    def parse_amount(raw: str) -> Optional[float]:
        """Parsea un string de monto a float."""
        if not raw:
            return None
        # Limpiar: quitar puntos de miles, convertir coma decimal
        cleaned = raw.strip()
        # Formato 1.200,00 → 1200.00
        if re.search(r'\d\.\d{3}', cleaned):
            cleaned = cleaned.replace('.', '').replace(',', '.')
        # Formato 1,200.00 → 1200.00
        elif re.search(r'\d,\d{3}', cleaned):
            cleaned = cleaned.replace(',', '')
        # Solo coma decimal: 1200,00 → 1200.00
        elif ',' in cleaned and '.' not in cleaned:
            cleaned = cleaned.replace(',', '.')
        # Solo puntos: 1200.00 → 1200.00
        else:
            cleaned = cleaned.replace(',', '')
        try:
            val = float(cleaned)
            return val if val > 0 else None
        except Exception:
            return None

    # Total — múltiples formatos latinoamericanos
    total_patterns = [
        # Formatos explícitos con label
        r'TOTAL\s+S/\.?\s*([\d,\.]{3,})',
        r'TOTAL\s+S/\s*([\d,\.]{3,})',
        r'Total\s*[:\s]\s*\$?\s*([\d\.]{3,})',
        r'TOTAL\s*[:\s]\s*\$?\s*([\d\.]{3,})',
        r'IMPORTE\s+TOTAL\s*[:\s]\s*([\d,\.]{3,})',
        r'MONTO\s+TOTAL\s*[:\s]\s*([\d,\.]{3,})',
        r'Total\s+a\s+pagar\s*[:\s]\s*([\d,\.]{3,})',
        r'TOTAL\s+A\s+PAGAR\s*[:\s]\s*([\d,\.]{3,})',
        r'TOTAL\s+VENTA\s*[:\s]\s*([\d,\.]{3,})',
        r'TOTAL\s+FACTURA\s*[:\s]\s*([\d,\.]{3,})',
        r'TOTAL\s+BOLETA\s*[:\s]\s*([\d,\.]{3,})',
        # TOTAL SI (formato peruano)
        r'TOTAL\s+SI\s*[:\s]?\s*([\d,\.]{3,})',
        # Total seguido de número en la misma línea (sin dos puntos)
        r'^\s*Total\s+(\d[\d\.]{2,})\s*$',
        r'^\s*TOTAL\s+(\d[\d\.]{2,})\s*$',
        # Formato tabla: Total al final de línea con espacios
        r'Total[\s\t]+(\d[\d\.]{2,})\s*$',
        r'TOTAL[\s\t]+(\d[\d\.]{2,})\s*$',
        # Monto con $ al final de línea (boletas chilenas)
        r'Total[:\s]*\$\s*(\d[\d\.]{2,})',
        r'TOTAL[:\s]*\$\s*(\d[\d\.]{2,})',
    ]
    for p in total_patterns:
        m = re.search(p, text, re.I | re.MULTILINE)
        if m:
            val = parse_amount(m.group(1))
            if val and val > 10:
                fields["monto_total"] = val
                break
    # Fallback: si no se encontró total, buscar el número más grande en líneas con 'total'
    if "monto_total" not in fields:
        for line in text.split('\n'):
            if re.search(r'total', line, re.I):
                nums = re.findall(r'[\d][\d\.]{2,}', line)
                for n in nums:
                    val = parse_amount(n)
                    if val and val > 100:
                        fields["monto_total"] = val
                        break
            if "monto_total" in fields:
                break

    # IVA — buscar en línea propia con el label "Iva:" o "IVA:"
    # Evitar capturar "119" de "producto iva 119"
    iva_patterns = [
        r'(?:^|\n)\s*I\.?V\.?A\.?\s*[:\s]\s*\$?\s*([\d\.]{2,})\s*(?:\n|$)',
        r'(?:^|\n)\s*IGV\s*[:\s]\s*\$?\s*([\d,\.]{2,})\s*(?:\n|$)',
        r'(?:^|\n)\s*Impuesto\s*[:\s]\s*([\d,\.]{2,})\s*(?:\n|$)',
        # Fallback: IVA seguido de número al final de línea
        r'I\.?V\.?A\.?[:\s]+\$?\s*(\d{2,6})\s*$',
    ]
    for p in iva_patterns:
        m = re.search(p, text, re.I | re.MULTILINE)
        if m:
            val = parse_amount(m.group(1))
            if val and val > 1:
                fields["iva"] = val
                break

    # Monto neto
    neto_patterns = [
        r'Monto\s+Neto\s*[:\s]\s*\$?\s*([\d\.]{3,})',
        r'MONTO\s+NETO\s*[:\s]\s*\$?\s*([\d\.]{3,})',
        r'Base\s+Imponible\s*[:\s]\s*([\d,\.]{3,})',
        r'Subtotal\s*[:\s]\s*([\d,\.]{3,})',
    ]
    for p in neto_patterns:
        m = re.search(p, text, re.I)
        if m:
            val = parse_amount(m.group(1))
            if val and val > 10:
                fields["monto_neto"] = val
                break

    # Calcular neto si tenemos total e IVA
    if "monto_total" in fields and "iva" in fields and "monto_neto" not in fields:
        neto = fields["monto_total"] - fields["iva"]
        if neto > 0:
            fields["monto_neto"] = round(neto, 2)

    # ── Proveedor ──
    prov_patterns = [
        # Después de la hora (HH:MM:SS) — el nombre de la empresa suele estar justo después
        r'\d{2}:\d{2}:\d{2}\s*[\n\r]+([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñÁÉÍÓÚÑ\s]{5,60}(?:SPA|S\.A\.|LTDA|SRL|SAC|EIRL|SAS)?)',
        # Inicio de línea con empresa (Productos "...", etc.)
        r'^(Productos\s+"[^"]+"|Productos\s+[A-Za-z]+)',
        # Después de "De:"
        r'De:\s*([^\n\r]{5,60})',
        # Nombre con sufijo empresarial explícito
        r'([A-ZÁÉÍÓÚÑ][A-Za-záéíóúñÁÉÍÓÚÑ\s]{5,50}(?:SPA|S\.A\.|LTDA|SRL|SAC|EIRL|SAS))',
    ]
    for pat in prov_patterns:
        for m in re.finditer(pat, text, re.I | re.MULTILINE):
            prov = m.group(1).strip().rstrip('"').strip()
            # Filtrar ruido OCR: no debe empezar con letras sueltas (S, N, etc.)
            # ni contener solo mayúsculas de 1-2 chars
            if (5 < len(prov) < 80 and
                    not re.match(r'^[A-Z]\s+[A-Z]', prov) and
                    not re.match(r'^\d', prov) and
                    not re.match(r'^(SN|NS|SS|NN)\s', prov)):
                fields["proveedor"] = prov
                break
        if fields.get("proveedor"):
            break

    # ── Descripción ──
    desc_patterns = [
        r'(?:Giro|GIRO)\s*[:\s]\s*([^\n\r]{5,100})',
        r'(?:DESCRIPCI[ÓO]N|DETALLE|CONCEPTO|GLOSA)\s*[:\s]\s*([^\n\r]{5,80})',
        r'(?:Descripci[oó]n|Detalle)\s*[:\s]\s*([^\n\r]{5,80})',
    ]
    for pat in desc_patterns:
        m = re.search(pat, text, re.I)
        if m:
            desc = m.group(1).strip()
            if len(desc) > 5:
                fields["descripcion"] = desc[:150]
                break

    # ── Moneda ──
    if re.search(r'S/\.?\s*\d|soles?|\bPEN\b', text, re.I):
        fields["moneda"] = "PEN"
    elif re.search(r'\bCLP\b|\$\s*\d|pesos?\s+chile', text, re.I):
        fields["moneda"] = "CLP"
    elif re.search(r'\bMXN\b|pesos?\s+mexic', text, re.I):
        fields["moneda"] = "MXN"
    elif re.search(r'\bUSD\b|\bUS\$\b|\bdolares?\b', text, re.I):
        fields["moneda"] = "USD"
    elif re.search(r'\bARS\b|pesos?\s+argent', text, re.I):
        fields["moneda"] = "ARS"

    return fields


# ── Prompts para LLM ──────────────────────────────────────────────────────────

EXTRACTION_PROMPT = """Extrae datos de este documento contable y devuelve JSON.

REGLAS IMPORTANTES:
- monto_total: busca la palabra TOTAL o Total seguida de un número. Ese número es el monto total.
- monto_neto: busca Neto, Monto Neto, Subtotal o Base Imponible.
- iva: busca IVA, I.V.A., IGV seguido de un número.
- Si ves "Total: 238" entonces monto_total=238.
- Si ves "Total 238" entonces monto_total=238.
- Los montos son SOLO números, sin $, sin puntos de miles, sin texto.
- rut_proveedor: formato XX.XXX.XXX-X o XXXXXXXX-X.
- fecha_emision: formato YYYY-MM-DD.

TEXTO:
{text}

Devuelve SOLO este JSON (sin texto adicional, sin markdown):
{{"tipo_documento":"FACTURA|BOLETA|RECIBO|NOTA_CREDITO|OTRO","proveedor":"nombre emisor o null","rut_proveedor":"RUT o null","fecha_emision":"YYYY-MM-DD o null","folio":"numero o null","descripcion":"descripcion o null","monto_neto":numero_o_null,"iva":numero_o_null,"monto_total":numero_o_null,"moneda":"CLP"}}"""

VISION_EXTRACTION_PROMPT = """Eres un experto en documentos contables latinoamericanos. Analiza esta imagen de un documento (factura, boleta, recibo) y extrae toda la información visible.

Extrae en formato JSON:
{{
  "tipo_documento": "FACTURA|BOLETA|RECIBO|NOTA_CREDITO|OTRO",
  "proveedor": "nombre del emisor",
  "rut_proveedor": "RUT/RUC del emisor",
  "fecha_emision": "YYYY-MM-DD",
  "folio": "número de documento",
  "descripcion": "descripción del producto/servicio",
  "monto_neto": número,
  "iva": número,
  "monto_total": número,
  "moneda": "CLP|USD|EUR|PEN|MXN|ARS",
  "texto_completo": "todo el texto visible en el documento"
}}

Responde SOLO con el JSON."""

CLASSIFICATION_PROMPT = """Eres un contador experto. Clasifica este gasto empresarial en la categoría contable más apropiada.

INFORMACIÓN DEL GASTO:
- Proveedor: {proveedor}
- Descripción: {descripcion}
- Tipo documento: {tipo_documento}
- Monto: {monto_total} {moneda}

CATEGORÍAS DISPONIBLES:
{categorias}

Responde en JSON:
{{
  "categoria": "nombre exacto de la categoría",
  "confianza": 0.0 a 1.0,
  "razon": "explicación breve de por qué esta categoría"
}}

Responde SOLO con el JSON."""

KEYWORD_MAP = {
    "tecnología": ["software", "hosting", "cloud", "internet", "computaci",
                   "informátic", "digital", "web", "servidor", "licencia",
                   "app", "sistema", "equipos computacionales", "programas"],
    "marketing": ["publicidad", "marketing", "diseño", "imprenta",
                  "redes sociales", "campaña", "flyer", "banner", "fotografía"],
    "operaciones": ["arriendo", "arrend", "servicio", "mantenci",
                    "limpieza", "seguridad", "agua", "luz", "gas", "electricidad"],
    "sueldos": ["sueldo", "salario", "honorario", "remuneraci", "liquidaci"],
    "impuestos": ["impuesto", "iva", "sii", "contribuci", "patente"],
    "transporte": ["transporte", "flete", "courier", "envío",
                   "despacho", "taxi", "uber", "combustible", "bencina"],
    "alimentación": ["restaurant", "comida", "almuerzo", "café",
                     "aliment", "supermercado", "catering"],
    "equipamiento": ["equipo", "maquinaria", "herramienta",
                     "mueble", "escritorio", "silla"],
    "servicios profesionales": ["consultoría", "asesoría", "abogado",
                                "contador", "auditor", "ingeniero", "notario"],
    "financiero": ["banco", "comisión", "interés", "crédito", "préstamo",
                   "leasing", "seguro", "afp", "isapre"],
}


def _parse_json_response(text: str) -> Dict:
    """Extrae JSON de la respuesta del LLM."""
    if not text:
        return {}
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"(\{[\s\S]*\})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    try:
        return json.loads(text)
    except Exception:
        pass
    return {}


# ── Pipeline Principal ────────────────────────────────────────────────────────

class DocumentPipelineAgent:
    """
    Agente de procesamiento de documentos con pipeline visible.
    Emite eventos SSE por cada paso del pipeline.
    """

    def __init__(self, db, empresa_id: str, uploads_dir: Path):
        self.db = db
        self.empresa_id = empresa_id
        self.uploads_dir = uploads_dir
        self._available_models: Optional[List[str]] = None
        self._text_model: Optional[str] = None
        self._vision_model: Optional[str] = None

    def _get_models(self):
        """Detecta modelos disponibles (cached)."""
        if self._available_models is None:
            self._available_models = _ollama_list()
            self._text_model = _find_best_model(TEXT_MODELS, self._available_models)
            self._vision_model = _find_best_model(VISION_MODELS, self._available_models)
            logger.info(f"Modelos — texto: {self._text_model}, visión: {self._vision_model}")
        return self._text_model, self._vision_model

    def _emit(self, step: str, status: str, data: Dict) -> str:
        """Genera un evento SSE."""
        event = {
            "step": step,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            **data
        }
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def process_stream(self, file_path: str, original_filename: str) -> Generator[str, None, None]:
        """
        Procesa un documento y emite eventos SSE por cada paso.
        Yields: strings con formato SSE.
        """
        doc_id = str(uuid.uuid4())
        ext = Path(file_path).suffix.lower()
        text_model, vision_model = self._get_models()

        # Contexto acumulado durante el pipeline
        ctx: Dict = {
            "doc_id": doc_id,
            "file_path": file_path,
            "original_filename": original_filename,
            "ocr_text": "",
            "vision_text": "",
            "combined_text": "",
            "fields": {},
            "classification": {},
            "audit": {},
            "saved": False,
        }

        # ── PASO 1: OCR ───────────────────────────────────────────────────────
        yield self._emit("ocr", "running", {
            "title": "Extracción de texto (OCR)",
            "message": f"Procesando {original_filename} con Tesseract multi-PSM..."
        })

        try:
            if ext == ".pdf":
                ocr_result = ocr_pdf(file_path)
            else:
                ocr_result = ocr_image(file_path)

            ctx["ocr_text"] = ocr_result["text"]
            ctx["combined_text"] = ocr_result["text"]

            yield self._emit("ocr", "done", {
                "title": "OCR completado",
                "message": f"Extraídas {ocr_result['words']} palabras via {ocr_result['method']}",
                "words": ocr_result["words"],
                "method": ocr_result["method"],
                "preview": ocr_result["text"][:400] if ocr_result["text"] else ""
            })
        except Exception as e:
            logger.error(f"OCR error: {e}")
            yield self._emit("ocr", "error", {
                "title": "OCR falló",
                "message": str(e)
            })

        # ── PASO 2: Visión VLLM ───────────────────────────────────────────────
        if vision_model and ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            yield self._emit("vision", "running", {
                "title": f"Análisis visual ({vision_model})",
                "message": f"Enviando imagen al modelo de visión {vision_model}..."
            })
            try:
                from PIL import Image
                import io as _io
                img = Image.open(file_path)
                if max(img.size) > 1600:
                    img.thumbnail((1600, 1600), Image.LANCZOS)
                buf = _io.BytesIO()
                img.convert("RGB").save(buf, format="JPEG", quality=85)
                img_b64 = base64.b64encode(buf.getvalue()).decode()

                vision_resp = _ollama_generate(
                    vision_model,
                    VISION_EXTRACTION_PROMPT,
                    image_b64=img_b64,
                    timeout=180
                )

                vision_fields = _parse_json_response(vision_resp)
                if vision_fields.get("texto_completo"):
                    ctx["vision_text"] = vision_fields.pop("texto_completo", "")
                    ctx["combined_text"] = (
                        ctx["ocr_text"] + "\n\n[VISIÓN]\n" + ctx["vision_text"]
                    ).strip()

                # Merge campos del vision (solo si no están vacíos)
                for k, v in vision_fields.items():
                    if v and v not in (None, "", "null", "None") and k not in ctx["fields"]:
                        ctx["fields"][k] = v

                yield self._emit("vision", "done", {
                    "title": f"Visión completada ({vision_model})",
                    "message": f"Campos adicionales: {[k for k, v in vision_fields.items() if v]}",
                    "fields_found": [k for k, v in vision_fields.items() if v],
                    "preview": ctx["vision_text"][:300] if ctx["vision_text"] else ""
                })
            except Exception as e:
                yield self._emit("vision", "error", {
                    "title": "Visión falló (continuando sin ella)",
                    "message": str(e)[:200]
                })
        else:
            reason = "No hay modelo de visión disponible" if not vision_model else "Archivo PDF — usando solo OCR"
            yield self._emit("vision", "info", {
                "title": "Visión omitida",
                "message": reason
            })

        # ── PASO 3: Extracción de campos ──────────────────────────────────────
        yield self._emit("extraction", "running", {
            "title": f"Extracción de campos ({text_model or 'regex'})",
            "message": "Identificando proveedor, fecha, montos, folio..."
        })

        try:
            # Regex siempre como base (rápido y confiable)
            regex_fields = _extract_by_regex(ctx["combined_text"])
            ctx["fields"].update({k: v for k, v in regex_fields.items()
                                   if k not in ctx["fields"] or not ctx["fields"][k]})

            # LLM si disponible (mejora campos no detectados por regex)
            if text_model and ctx["combined_text"] and len(ctx["combined_text"].split()) >= 10:
                text_to_analyze = ctx["combined_text"][:3000]
                llm_resp = _ollama_generate(
                    text_model,
                    EXTRACTION_PROMPT.format(text=text_to_analyze),
                    timeout=90
                )
                llm_fields = _parse_json_response(llm_resp)
                # LLM tiene prioridad para monto_total y campos no encontrados por regex
                PRIORITY_FIELDS = {"monto_total", "monto_neto", "iva", "proveedor", "descripcion"}
                for k, v in llm_fields.items():
                    if v is not None and v not in ("", "null", "None"):
                        # Para campos de monto y proveedor, LLM siempre tiene prioridad
                        if k in PRIORITY_FIELDS:
                            ctx["fields"][k] = v
                        elif k not in ctx["fields"] or not ctx["fields"][k]:
                            ctx["fields"][k] = v

            # Limpiar nulls y strings vacíos
            ctx["fields"] = {
                k: v for k, v in ctx["fields"].items()
                if v is not None and v != "" and str(v) not in ("null", "None")
            }

            yield self._emit("extraction", "done", {
                "title": "Campos extraídos",
                "message": f"Encontrados {len(ctx['fields'])} campos",
                "fields": {k: str(v) for k, v in ctx["fields"].items()}
            })
        except Exception as e:
            logger.error(f"Extraction error: {e}")
            yield self._emit("extraction", "error", {
                "title": "Extracción falló",
                "message": str(e),
                "fields": {k: str(v) for k, v in ctx["fields"].items()}
            })

        # ── PASO 4: Clasificación ─────────────────────────────────────────────
        yield self._emit("classification", "running", {
            "title": f"Clasificación del gasto ({text_model or 'keywords'})",
            "message": "Determinando categoría contable..."
        })

        try:
            from models import CategoriaContable
            categorias = self.db.query(CategoriaContable).filter(
                CategoriaContable.empresa_id == self.empresa_id
            ).all()
            # CategoriaContable no tiene campo 'descripcion' — usar tipo_gasto
            cat_list = [{"id": c.id, "nombre": c.nombre,
                         "descripcion": c.tipo_gasto or c.nombre} for c in categorias]

            classification: Dict = {
                "categoria_sugerida": None, "confianza": 0.0,
                "razon": "", "categoria_id": None
            }

            if cat_list:
                combined_lower = ctx["combined_text"].lower()
                proveedor_lower = str(ctx["fields"].get("proveedor", "")).lower()
                desc_lower = str(ctx["fields"].get("descripcion", "")).lower()
                search_text = combined_lower + " " + proveedor_lower + " " + desc_lower

                # 1. LLM classification
                if text_model:
                    cat_names = "\n".join(
                        f"- {c['nombre']}" + (f" ({c['descripcion']})" if c['descripcion'] != c['nombre'] else "")
                        for c in cat_list
                    )
                    llm_resp = _ollama_generate(
                        text_model,
                        CLASSIFICATION_PROMPT.format(
                            proveedor=ctx["fields"].get("proveedor", "Desconocido"),
                            descripcion=ctx["fields"].get("descripcion",
                                         ctx["combined_text"][:200]),
                            tipo_documento=ctx["fields"].get("tipo_documento", ""),
                            monto_total=ctx["fields"].get("monto_total", ""),
                            moneda=ctx["fields"].get("moneda", "CLP"),
                            categorias=cat_names
                        ),
                        timeout=60
                    )
                    cls_data = _parse_json_response(llm_resp)
                    if cls_data.get("categoria"):
                        classification["categoria_sugerida"] = cls_data["categoria"]
                        classification["confianza"] = float(cls_data.get("confianza", 0.7))
                        classification["razon"] = cls_data.get("razon", "")

                # 2. Keyword fallback si LLM no clasificó
                if not classification["categoria_sugerida"]:
                    for cat in cat_list:
                        cat_name_lower = cat["nombre"].lower()
                        for key, keywords in KEYWORD_MAP.items():
                            if key in cat_name_lower:
                                for kw in keywords:
                                    if kw in search_text:
                                        classification["categoria_sugerida"] = cat["nombre"]
                                        classification["confianza"] = 0.65
                                        classification["razon"] = f"Palabra clave '{kw}' encontrada"
                                        break
                            if classification["categoria_sugerida"]:
                                break
                        if classification["categoria_sugerida"]:
                            break

                # 3. Buscar ID de la categoría sugerida
                if classification["categoria_sugerida"]:
                    for c in cat_list:
                        if c["nombre"].lower() == classification["categoria_sugerida"].lower():
                            classification["categoria_id"] = c["id"]
                            break
                    # Si no hay match exacto, buscar parcial
                    if not classification["categoria_id"]:
                        for c in cat_list:
                            if (c["nombre"].lower() in classification["categoria_sugerida"].lower() or
                                    classification["categoria_sugerida"].lower() in c["nombre"].lower()):
                                classification["categoria_id"] = c["id"]
                                classification["categoria_sugerida"] = c["nombre"]
                                break

            ctx["classification"] = classification

            yield self._emit("classification", "done", {
                "title": "Clasificación completada",
                "message": (f"Categoría: {classification['categoria_sugerida'] or 'Sin categoría'} "
                            f"(confianza: {int(classification['confianza'] * 100)}%)"),
                "categoria_sugerida": classification["categoria_sugerida"],
                "confianza": classification["confianza"],
                "razon": classification["razon"],
                "categorias_disponibles": [c["nombre"] for c in cat_list]
            })
        except Exception as e:
            logger.error(f"Classification error: {e}")
            yield self._emit("classification", "error", {
                "title": "Clasificación falló",
                "message": str(e)
            })

        # ── PASO 5: Auditoría ─────────────────────────────────────────────────
        yield self._emit("audit", "running", {
            "title": "Auditoría y validación",
            "message": "Verificando duplicados y anomalías..."
        })

        try:
            from models import Documento as DocModel
            excepciones = []
            requiere_revision = False

            # Hash del archivo
            sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            file_hash = sha256.hexdigest()

            # Verificar duplicado
            dup = self.db.query(DocModel).filter(
                DocModel.empresa_id == self.empresa_id,
                DocModel.hash_documento == file_hash
            ).first()
            if dup:
                excepciones.append({
                    "tipo": "DUPLICADO",
                    "mensaje": f"Documento ya existe (ID: {dup.id[:8]}...)"
                })
                requiere_revision = True

            # Validar campos críticos
            if not ctx["fields"].get("monto_total"):
                excepciones.append({"tipo": "CAMPO_FALTANTE",
                                     "mensaje": "Monto total no detectado"})
                requiere_revision = True

            if not ctx["fields"].get("proveedor"):
                excepciones.append({"tipo": "CAMPO_FALTANTE",
                                     "mensaje": "Proveedor no detectado"})
                requiere_revision = True

            # Baja confianza de clasificación
            if ctx["classification"].get("confianza", 0) < 0.5:
                excepciones.append({"tipo": "BAJA_CONFIANZA",
                                     "mensaje": "Clasificación con baja confianza"})
                requiere_revision = True

            ctx["audit"] = {
                "hash": file_hash,
                "excepciones": excepciones,
                "requiere_revision": requiere_revision
            }

            yield self._emit("audit", "done", {
                "title": "Auditoría completada",
                "message": (f"{len(excepciones)} alertas encontradas"
                            if excepciones else "Sin alertas — documento válido"),
                "excepciones": excepciones,
                "requiere_revision": requiere_revision
            })
        except Exception as e:
            ctx["audit"] = {"hash": "", "excepciones": [], "requiere_revision": True}
            yield self._emit("audit", "error", {
                "title": "Auditoría falló",
                "message": str(e)
            })

        # ── PASO 6: Guardado ──────────────────────────────────────────────────
        yield self._emit("save", "running", {
            "title": "Guardando documento",
            "message": "Persistiendo en base de datos..."
        })

        try:
            saved_doc = self._save_document(
                doc_id=doc_id,
                file_path=file_path,
                original_filename=original_filename,
                ctx=ctx
            )
            ctx["saved"] = True

            yield self._emit("save", "done", {
                "title": "Documento guardado",
                "message": f"ID: {saved_doc.id[:8]}...",
                "doc_id": saved_doc.id,
                "empresa_id": self.empresa_id
            })

            # Evento final con resumen completo
            yield self._emit("complete", "done", {
                "title": "Procesamiento completado",
                "message": "El documento ha sido procesado y guardado exitosamente",
                "doc_id": saved_doc.id,
                "fields": {k: str(v) for k, v in ctx["fields"].items()},
                "classification": ctx["classification"],
                "audit": ctx["audit"],
                "ocr_words": len([w for w in ctx["combined_text"].split() if len(w) > 2]),
                "models_used": {
                    "text": text_model,
                    "vision": vision_model
                }
            })

        except Exception as e:
            logger.error(f"Save error: {e}")
            # Si es un error de duplicado (UNIQUE constraint), devolver el documento existente
            err_str = str(e)
            if "UNIQUE constraint" in err_str and "hash_documento" in err_str:
                # Buscar el documento existente por hash
                try:
                    from models import Documento as DocModel
                    # Rollback the failed transaction before querying
                    self.db.rollback()
                    file_hash = ctx.get("audit", {}).get("hash", "")
                    existing = None
                    if file_hash:
                        existing = self.db.query(DocModel).filter(
                            DocModel.empresa_id == self.empresa_id,
                            DocModel.hash_documento == file_hash
                        ).first()
                    if existing:
                        yield self._emit("save", "info", {
                            "title": "Documento duplicado detectado",
                            "message": f"Este documento ya fue procesado (ID: {existing.id[:8]}...)",
                            "doc_id": existing.id
                        })
                        yield self._emit("complete", "done", {
                            "title": "Documento ya existente",
                            "message": "Este documento ya fue procesado anteriormente",
                            "doc_id": existing.id,
                            "fields": {k: str(v) for k, v in ctx["fields"].items()},
                            "classification": ctx["classification"],
                            "audit": ctx["audit"],
                            "ocr_words": len([w for w in ctx["combined_text"].split() if len(w) > 2]),
                            "models_used": {"text": text_model, "vision": vision_model},
                            "duplicate": True
                        })
                        return
                except Exception as e2:
                    logger.error(f"Duplicate lookup error: {e2}")
            yield self._emit("save", "error", {
                "title": "Error al guardar",
                "message": err_str[:200]
            })

    def _save_document(self, doc_id: str, file_path: str,
                       original_filename: str, ctx: Dict):
        """Guarda el documento en la base de datos con los campos correctos del modelo."""
        from models import (Documento, TipoDocumento, EstadoRevision)

        fields = ctx["fields"]
        classification = ctx["classification"]
        audit = ctx["audit"]

        # Tipo de documento
        tipo_str = str(fields.get("tipo_documento", "OTRO")).upper()
        tipo_str = re.sub(r'\s+', '_', tipo_str)
        tipo_str = re.sub(r'[ÁÉÍÓÚ]', lambda m: {'Á':'A','É':'E','Í':'I','Ó':'O','Ú':'U'}.get(m.group(), m.group()), tipo_str)
        tipo_map = {
            "FACTURA": TipoDocumento.FACTURA,
            "FACTURA_ELECTRONICA": TipoDocumento.FACTURA,
            "BOLETA": TipoDocumento.BOLETA,
            "BOLETA_ELECTRONICA": TipoDocumento.BOLETA,
            "BOLETA_DE_VENTA": TipoDocumento.BOLETA,
            "RECIBO": TipoDocumento.COMPROBANTE,
            "RECIBO_DE_HONORARIOS": TipoDocumento.COMPROBANTE,
            "NOTA_CREDITO": TipoDocumento.COMPROBANTE,
            "NOTA_DE_CREDITO": TipoDocumento.COMPROBANTE,
            "COMPROBANTE": TipoDocumento.COMPROBANTE,
            "CARTOLA": TipoDocumento.CARTOLA,
        }
        tipo_enum = tipo_map.get(tipo_str, TipoDocumento.OTRO)

        # Fecha
        fecha_emision = None
        if fields.get("fecha_emision"):
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]:
                try:
                    fecha_emision = datetime.strptime(str(fields["fecha_emision"]), fmt)
                    break
                except Exception:
                    pass

        # Montos
        def to_float(v) -> Optional[float]:
            if v is None:
                return None
            try:
                return float(str(v).replace(",", ".").replace(" ", ""))
            except Exception:
                return None

        # Estado
        estado = (EstadoRevision.PENDIENTE
                  if audit.get("requiere_revision", True)
                  else EstadoRevision.REVISADO)

        # Hash único — si ya existe, usar un hash diferente para evitar conflicto
        file_hash = audit.get("hash", "")
        if not file_hash:
            file_hash = str(uuid.uuid4()).replace("-", "")

        doc = Documento(
            id=doc_id,
            empresa_id=self.empresa_id,
            tipo_documento=tipo_enum,
            proveedor=fields.get("proveedor"),
            rut_proveedor=fields.get("rut_proveedor"),
            fecha_emision=fecha_emision,
            folio=str(fields.get("folio", "")) if fields.get("folio") else None,
            monto_neto=to_float(fields.get("monto_neto")),
            iva=to_float(fields.get("iva")),
            monto_total=to_float(fields.get("monto_total")),
            moneda=str(fields.get("moneda", "CLP")),
            categoria_id=classification.get("categoria_id"),
            categoria_sugerida=classification.get("categoria_sugerida"),
            confianza_clasificacion=float(classification.get("confianza", 0.0)),
            estado_revision=estado,
            ruta_archivo_original=str(file_path),
            hash_documento=file_hash,
            texto_extraido=ctx["combined_text"][:5000] if ctx["combined_text"] else "",
            campos_extraidos=json.dumps(fields, ensure_ascii=False),
            excepciones=json.dumps(audit.get("excepciones", []), ensure_ascii=False),
        )

        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc
