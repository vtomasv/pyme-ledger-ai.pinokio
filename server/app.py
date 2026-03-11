"""
Servidor principal del plugin pyme-ledger-ai.
API FastAPI con endpoints para empresas, documentos, analítica y exportación.
"""
import os
import json
import uuid
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse, StreamingResponse, Response
from pydantic import BaseModel
import uvicorn
from sqlalchemy.orm import Session

# Importar módulos
from database import init_db, get_db, SessionLocal, save_json, load_json
from models import (
    Empresa, CentroCosto, CategoriaContable, Documento, 
    TipoDocumento, EstadoRevision
)
from orchestration import DocumentProcessingPipeline
from pipeline_agent import DocumentPipelineAgent
from analytics import AnalyticsEngine
from analytics.recommender import get_recommender_engine
from analytics.exporter import get_export_engine

# Configuración — rutas absolutas desde __file__ (requerido por Pinokio)
# server/app.py → parent = server/ → parent.parent = raíz del plugin
BASE_DIR = Path(__file__).parent.parent.resolve()
PORT = int(os.environ.get("PORT", 8000))
DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DIR / "data")))
PLUGIN_DIR = Path(os.environ.get("PLUGIN_DIR", str(BASE_DIR)))
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

print(f"INFO: BASE_DIR  = {BASE_DIR}")
print(f"INFO: DATA_DIR  = {DATA_DIR}")
print(f"INFO: PORT      = {PORT}")

# Inicializar BD
init_db()

app = FastAPI(title="Pyme Ledger AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Modelos Pydantic
# ============================================================

class EmpresaCreate(BaseModel):
    razon_social: Optional[str] = None
    nombre: Optional[str] = None          # alias de razon_social (enviado por la UI)
    nombre_fantasia: Optional[str] = None
    rut: str = "00.000.000-0"
    pais: str = "Chile"
    moneda_base: str = "CLP"
    regimen_tributario: Optional[str] = None
    giro: Optional[str] = None

    def get_razon_social(self) -> str:
        return self.razon_social or self.nombre or "Empresa sin nombre"


class CentroCostoCreate(BaseModel):
    codigo: str
    nombre: str
    descripcion: Optional[str] = None


class CategoriaContableCreate(BaseModel):
    codigo: Optional[str] = None
    nombre: str
    tipo_gasto: Optional[str] = None
    tipo: Optional[str] = "gasto"
    descripcion: Optional[str] = None
    deducibilidad: str = "Total"
    regla_iva: str = "Recuperable"


class DocumentoUpdate(BaseModel):
    categoria_id: Optional[str] = None
    centro_costo_id: Optional[str] = None
    estado_revision: Optional[str] = None
    notas_revision: Optional[str] = None


# ============================================================
# Endpoints: Empresas
# ============================================================

@app.get("/")
async def root():
    """Redirecciona a la UI."""
    return RedirectResponse(url="/ui/index.html")


@app.post("/api/empresas")
async def create_empresa(empresa: EmpresaCreate):
    """Crea una nueva empresa."""
    db = SessionLocal()
    try:
        empresa_id = str(uuid.uuid4())
        razon_social = empresa.get_razon_social()
        new_empresa = Empresa(
            id=empresa_id,
            razon_social=razon_social,
            nombre_fantasia=empresa.nombre_fantasia or empresa.giro or razon_social,
            rut=empresa.rut,
            pais=empresa.pais,
            moneda_base=empresa.moneda_base,
            regimen_tributario=empresa.regimen_tributario or empresa.giro
        )
        db.add(new_empresa)
        db.commit()
        
        return {
            "id": empresa_id,
            "razon_social": razon_social,
            "nombre": razon_social,
            "rut": empresa.rut,
            "giro": empresa.giro,
            "mensaje": "Empresa creada exitosamente"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


@app.get("/api/empresas")
async def list_empresas():
    """Lista todas las empresas."""
    db = SessionLocal()
    try:
        empresas = db.query(Empresa).filter(Empresa.activa == True).all()
        return {
            "empresas": [
                {
                    "id": e.id,
                    "razon_social": e.razon_social,
                    "nombre_fantasia": e.nombre_fantasia,
                    "rut": e.rut,
                    "giro": e.regimen_tributario,
                    "pais": e.pais,
                    "moneda_base": e.moneda_base
                }
                for e in empresas
            ]
        }
    finally:
        db.close()


@app.get("/api/empresas/{empresa_id}")
async def get_empresa(empresa_id: str):
    """Obtiene detalles de una empresa."""
    db = SessionLocal()
    try:
        empresa = db.query(Empresa).filter(Empresa.id == empresa_id).first()
        if not empresa:
            raise HTTPException(status_code=404, detail="Empresa no encontrada")
        
        return {
            "id": empresa.id,
            "razon_social": empresa.razon_social,
            "nombre_fantasia": empresa.nombre_fantasia,
            "rut": empresa.rut,
            "pais": empresa.pais,
            "moneda_base": empresa.moneda_base,
            "regimen_tributario": empresa.regimen_tributario,
            "fecha_creacion": empresa.fecha_creacion.isoformat()
        }
    finally:
        db.close()


# ============================================================
# Endpoints: Centros de Costo
# ============================================================

@app.post("/api/empresas/{empresa_id}/centros-costo")
async def create_cost_center(empresa_id: str, cc: CentroCostoCreate):
    """Crea un nuevo centro de costo."""
    db = SessionLocal()
    try:
        # Verificar que la empresa existe
        empresa = db.query(Empresa).filter(Empresa.id == empresa_id).first()
        if not empresa:
            raise HTTPException(status_code=404, detail="Empresa no encontrada")
        
        cc_id = str(uuid.uuid4())
        new_cc = CentroCosto(
            id=cc_id,
            empresa_id=empresa_id,
            codigo=cc.codigo,
            nombre=cc.nombre,
            descripcion=cc.descripcion
        )
        db.add(new_cc)
        db.commit()
        
        return {"id": cc_id, "codigo": cc.codigo, "nombre": cc.nombre}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


@app.get("/api/empresas/{empresa_id}/centros-costo")
async def list_cost_centers(empresa_id: str):
    """Lista centros de costo de una empresa."""
    db = SessionLocal()
    try:
        centers = db.query(CentroCosto).filter(
            CentroCosto.empresa_id == empresa_id,
            CentroCosto.activo == True
        ).all()
        
        return {
            "centros_costo": [
                {
                    "id": c.id,
                    "codigo": c.codigo,
                    "nombre": c.nombre,
                    "descripcion": c.descripcion
                }
                for c in centers
            ]
        }
    finally:
        db.close()


# ============================================================
# Endpoints: Categorías Contables
# ============================================================

@app.post("/api/empresas/{empresa_id}/categorias")
async def create_category(empresa_id: str, cat: CategoriaContableCreate):
    """Crea una nueva categoría contable."""
    db = SessionLocal()
    try:
        empresa = db.query(Empresa).filter(Empresa.id == empresa_id).first()
        if not empresa:
            raise HTTPException(status_code=404, detail="Empresa no encontrada")
        
        cat_id = str(uuid.uuid4())
        new_cat = CategoriaContable(
            id=cat_id,
            empresa_id=empresa_id,
            codigo=cat.codigo or cat_id[:8],
            nombre=cat.nombre,
            tipo_gasto=cat.tipo_gasto or cat.tipo or "gasto",
            deducibilidad=cat.deducibilidad,
            regla_iva=cat.regla_iva
        )
        db.add(new_cat)
        db.commit()
        
        return {"id": cat_id, "codigo": cat.codigo or cat_id[:8], "nombre": cat.nombre}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


@app.get("/api/empresas/{empresa_id}/categorias")
async def list_categories(empresa_id: str):
    """Lista categorías contables de una empresa."""
    db = SessionLocal()
    try:
        categories = db.query(CategoriaContable).filter(
            CategoriaContable.empresa_id == empresa_id,
            CategoriaContable.activa == True
        ).all()
        
        return {
            "categorias": [
                {
                    "id": c.id,
                    "codigo": c.codigo,
                    "nombre": c.nombre,
                    "tipo_gasto": c.tipo_gasto,
                    "deducibilidad": c.deducibilidad
                }
                for c in categories
            ]
        }
    finally:
        db.close()


# ============================================================
# Endpoints: Documentos
# ============================================================

@app.post("/api/empresas/{empresa_id}/documentos/upload")
async def upload_document(empresa_id: str, file: UploadFile = File(...)):
    """
    Paso 1: Sube el archivo y lo guarda en disco.
    Retorna file_name (solo el nombre del archivo) para luego conectar al endpoint SSE.
    """
    db = SessionLocal()
    try:
        empresa = db.query(Empresa).filter(Empresa.id == empresa_id).first()
        if not empresa:
            raise HTTPException(status_code=404, detail="Empresa no encontrada")

        # Guardar archivo con nombre único
        original_name = file.filename or "documento"
        file_extension = Path(original_name).suffix.lower()
        if not file_extension or file_extension not in [".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]:
            file_extension = ".pdf"
        file_id = str(uuid.uuid4())
        file_name = f"{file_id}{file_extension}"
        file_path = UPLOADS_DIR / file_name

        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Archivo vacío")

        with open(str(file_path), "wb") as f:
            f.write(contents)

        print(f"INFO: Archivo guardado → {file_path} ({len(contents):,} bytes)")

        return {
            "status": "uploaded",
            "file_id": file_id,
            "file_name": file_name,
            "file_path": str(file_path),
            "original_filename": original_name,
            "size_bytes": len(contents),
            "empresa_id": empresa_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/api/empresas/{empresa_id}/documentos/process-stream")
async def process_document_stream(
    empresa_id: str,
    file_name: str = None,
    file_path: str = None,
    original_filename: str = "documento"
):
    """
    Paso 2: Procesa el archivo subido y emite eventos SSE por cada paso del pipeline.
    Acepta file_name (solo nombre del archivo) o file_path (ruta completa).
    Pasos: OCR → Visión VLLM → Extracción LLM → Clasificación → Auditoría → Guardado
    """
    db = SessionLocal()
    try:
        empresa = db.query(Empresa).filter(Empresa.id == empresa_id).first()
        if not empresa:
            raise HTTPException(status_code=404, detail="Empresa no encontrada")

        # Resolver ruta de forma segura — siempre dentro de UPLOADS_DIR
        if file_name:
            # Sanitizar: solo el nombre del archivo, sin separadores de directorio
            safe_name = Path(file_name).name  # elimina cualquier path traversal
            fp = UPLOADS_DIR / safe_name
        elif file_path:
            # Fallback: construir desde el nombre del archivo en la ruta
            fp = UPLOADS_DIR / Path(file_path).name
        else:
            raise HTTPException(status_code=400, detail="Se requiere file_name o file_path")

        if not fp.exists():
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado en uploads: {fp.name}")

        agent = DocumentPipelineAgent(db, empresa_id, UPLOADS_DIR)

        def generate():
            try:
                for event in agent.process_stream(str(fp), original_filename):
                    yield event
            except Exception as e:
                import json as _json
                yield f"data: {_json.dumps({'step': 'error', 'status': 'error', 'message': str(e)})}\n\n"
            finally:
                db.close()

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-store",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
                "Connection": "keep-alive",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        db.close()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/empresas/{empresa_id}/documentos")
async def list_documents(empresa_id: str, limit: int = 50, offset: int = 0):
    """Lista documentos de una empresa."""
    db = SessionLocal()
    try:
        docs = db.query(Documento).filter(
            Documento.empresa_id == empresa_id
        ).order_by(Documento.fecha_creacion.desc()).offset(offset).limit(limit).all()
        
        total = db.query(Documento).filter(
            Documento.empresa_id == empresa_id
        ).count()
        
        return {
            "total": total,
            "documentos": [
                {
                    "id": d.id,
                    "nombre_archivo": Path(d.ruta_archivo_original).name if d.ruta_archivo_original else (d.id + ".pdf"),
                    "tipo_archivo": Path(d.ruta_archivo_original).suffix.lower().lstrip('.') if d.ruta_archivo_original else "pdf",
                    "fecha_emision": d.fecha_emision.isoformat() if d.fecha_emision else None,
                    "fecha_subida": d.fecha_creacion.isoformat() if d.fecha_creacion else None,
                    "proveedor": d.proveedor,
                    "rut_proveedor": d.rut_proveedor,
                    "folio": d.folio,
                    "monto_neto": float(d.monto_neto) if d.monto_neto else None,
                    "iva": float(d.iva) if d.iva else None,
                    "monto_total": float(d.monto_total) if d.monto_total else None,
                    "tipo_documento": d.tipo_documento.value if d.tipo_documento else None,
                    "categoria_id": d.categoria_id,
                    "categoria_nombre": d.categoria.nombre if d.categoria else None,
                    "estado_revision": d.estado_revision.value if d.estado_revision else "pendiente",
                    "confianza_clasificacion": d.confianza_clasificacion,
                    "categoria_sugerida": d.categoria.nombre if d.categoria else None,
                    "razon_clasificacion": d.categoria_sugerida if d.categoria_sugerida else None,
                    "texto_extraido": d.texto_extraido[:300] if d.texto_extraido else None,
                    "excepciones": json.loads(d.excepciones) if d.excepciones else []
                }
                for d in docs
            ]
        }
    finally:
        db.close()


@app.get("/api/empresas/{empresa_id}/documentos/{doc_id}")
async def get_document(empresa_id: str, doc_id: str):
    """Obtiene detalles de un documento."""
    db = SessionLocal()
    try:
        doc = db.query(Documento).filter(
            Documento.id == doc_id,
            Documento.empresa_id == empresa_id
        ).first()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        
        # Parsear excepciones
        excepciones = []
        if doc.excepciones:
            try:
                excepciones = json.loads(doc.excepciones)
            except:
                pass
        
        # Determinar extensión para preview
        ruta = doc.ruta_archivo_original or ""
        ext = Path(ruta).suffix.lower().lstrip('.') if ruta else ""
        estado_val = doc.estado_revision.value.lower() if doc.estado_revision else "pendiente"
        return {
            "id": doc.id,
            "fecha": doc.fecha_emision.isoformat() if doc.fecha_emision else None,
            "fecha_emision": doc.fecha_emision.isoformat() if doc.fecha_emision else None,
            "tipo": doc.tipo_documento.value if doc.tipo_documento else None,
            "tipo_documento": doc.tipo_documento.value if doc.tipo_documento else None,
            "proveedor": doc.proveedor,
            "rut": doc.rut_proveedor,
            "rut_proveedor": doc.rut_proveedor,
            "folio": doc.folio,
            "monto_neto": doc.monto_neto,
            "iva": doc.iva,
            "monto_total": doc.monto_total,
            "moneda": doc.moneda,
            "categoria": doc.categoria.nombre if doc.categoria else None,
            "categoria_nombre": doc.categoria.nombre if doc.categoria else None,
            "categoria_id": doc.categoria_id,
            "centro_costo": doc.centro_costo.nombre if doc.centro_costo else None,
            "confianza": doc.confianza_clasificacion,
            "estado": estado_val,
            "estado_revision": estado_val,
            "excepciones": excepciones,
            "texto_extraido": doc.texto_extraido[:500] if doc.texto_extraido else None,
            "ruta_archivo_original": ruta,
            "tipo_archivo": ext,
            "nombre_archivo": Path(ruta).name if ruta else (doc.id + ".pdf")
        }
    finally:
        db.close()


@app.put("/api/empresas/{empresa_id}/documentos/{doc_id}")
async def update_document(empresa_id: str, doc_id: str, update: DocumentoUpdate):
    """Actualiza un documento."""
    db = SessionLocal()
    try:
        doc = db.query(Documento).filter(
            Documento.id == doc_id,
            Documento.empresa_id == empresa_id
        ).first()
        
        if not doc:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        
        if update.categoria_id:
            doc.categoria_id = update.categoria_id
        if update.centro_costo_id:
            doc.centro_costo_id = update.centro_costo_id
        if update.estado_revision:
            doc.estado_revision = EstadoRevision[update.estado_revision.upper()]
        if update.notas_revision:
            doc.notas_revision = update.notas_revision
        
        doc.fecha_revision = datetime.utcnow()
        db.commit()
        
        return {"mensaje": "Documento actualizado"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        db.close()


# ============================================================
# Endpoints: Analítica
# ============================================================

@app.get("/api/empresas/{empresa_id}/analytics/dashboard")
async def get_dashboard(empresa_id: str, period_days: int = 30):
    """Obtiene dashboard de analítica."""
    db = SessionLocal()
    try:
        analytics = AnalyticsEngine(db, empresa_id)
        return analytics.get_dashboard_summary(period_days)
    finally:
        db.close()


@app.get("/api/empresas/{empresa_id}/analytics/categorias")
async def get_categories_analysis(empresa_id: str, period_days: int = 30):
    """Análisis de gastos por categoría."""
    db = SessionLocal()
    try:
        analytics = AnalyticsEngine(db, empresa_id)
        return analytics.get_expenses_by_category(period_days)
    finally:
        db.close()


@app.get("/api/empresas/{empresa_id}/analytics/proveedores")
async def get_providers_analysis(empresa_id: str, period_days: int = 30):
    """Análisis de gastos por proveedor."""
    db = SessionLocal()
    try:
        analytics = AnalyticsEngine(db, empresa_id)
        return analytics.get_expenses_by_provider(period_days)
    finally:
        db.close()


@app.get("/api/empresas/{empresa_id}/analytics/tendencias")
async def get_trends(empresa_id: str, months: int = 12):
    """Análisis de tendencias mensuales."""
    db = SessionLocal()
    try:
        analytics = AnalyticsEngine(db, empresa_id)
        return analytics.get_monthly_trend(months)
    finally:
        db.close()


@app.get("/api/empresas/{empresa_id}/analytics/alertas")
async def get_alerts(empresa_id: str, period_days: int = 30):
    """Obtiene alertas y anomalías."""
    db = SessionLocal()
    try:
        analytics = AnalyticsEngine(db, empresa_id)
        return analytics.get_anomalies_and_alerts(period_days)
    finally:
        db.close()


# ============================================================
# Endpoints: Recomendaciones
# ============================================================

@app.get("/api/empresas/{empresa_id}/recomendaciones")
async def get_recommendations(empresa_id: str, period_days: int = 90):
    """Obtiene recomendaciones de optimización."""
    db = SessionLocal()
    try:
        recommender = get_recommender_engine(db, empresa_id)
        return recommender.get_recommendations(period_days)
    finally:
        db.close()


# ============================================================
# Endpoints: Exportación
# ============================================================

@app.post("/api/empresas/{empresa_id}/export/csv")
async def export_csv(empresa_id: str, period_days: int = 30):
    """Exporta documentos a CSV."""
    db = SessionLocal()
    try:
        exporter = get_export_engine(db, empresa_id, str(DATA_DIR))
        result = exporter.export_to_csv(period_days)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    finally:
        db.close()


@app.post("/api/empresas/{empresa_id}/export/pdf")
async def export_pdf(empresa_id: str, period_days: int = 30):
    """Exporta reporte a PDF."""
    db = SessionLocal()
    try:
        # Obtener datos de analítica
        analytics = AnalyticsEngine(db, empresa_id)
        analytics_data = analytics.get_dashboard_summary(period_days)
        analytics_data.update(analytics.get_expenses_by_category(period_days))
        
        # Exportar a PDF
        exporter = get_export_engine(db, empresa_id, str(DATA_DIR))
        result = exporter.export_to_pdf(analytics_data)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    finally:
        db.close()


@app.post("/api/empresas/{empresa_id}/export/libro-contable")
async def export_ledger(empresa_id: str, period_days: int = 30):
    """Exporta libro contable."""
    db = SessionLocal()
    try:
        exporter = get_export_engine(db, empresa_id, str(DATA_DIR))
        result = exporter.export_accounting_ledger(period_days)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
    finally:
        db.close()


@app.get("/api/exports/{filename}")
async def download_export(filename: str):
    """Descarga un archivo exportado."""
    file_path = DATA_DIR / "exports" / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/octet-stream"
    )


# ============================================================
# Endpoints: Preview y descarga de documentos
# ============================================================

@app.get("/api/documentos/{doc_id}/preview")
async def preview_document(doc_id: str):
    """Devuelve la imagen de preview de un documento (imagen directa o thumbnail de PDF)."""
    import io
    from fastapi.responses import StreamingResponse, Response
    db = SessionLocal()
    try:
        doc = db.query(Documento).filter(Documento.id == doc_id).first()
        if not doc or not doc.ruta_archivo_original:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        file_path = Path(doc.ruta_archivo_original)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {file_path}")
        ext = file_path.suffix.lower()
        headers = {"Cache-Control": "max-age=3600", "Access-Control-Allow-Origin": "*"}

        # Imágenes: servir directamente
        if ext in (".jpg", ".jpeg"):
            with open(str(file_path), "rb") as f:
                data = f.read()
            return Response(content=data, media_type="image/jpeg", headers=headers)
        elif ext == ".png":
            with open(str(file_path), "rb") as f:
                data = f.read()
            return Response(content=data, media_type="image/png", headers=headers)
        elif ext == ".pdf":
            # Intentar thumbnail con pdf2image
            try:
                from pdf2image import convert_from_path
                pages = convert_from_path(str(file_path), first_page=1, last_page=1, dpi=150)
                if pages:
                    img_io = io.BytesIO()
                    pages[0].save(img_io, format="JPEG", quality=80)
                    img_io.seek(0)
                    return StreamingResponse(img_io, media_type="image/jpeg", headers=headers)
            except Exception as e:
                print(f"INFO: pdf2image no disponible para preview ({e})")
            # Fallback: retornar 204 para que la UI muestre ícono de PDF
            return Response(status_code=204, headers=headers)
        else:
            return Response(status_code=204, headers=headers)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en preview: {e}")
        return Response(status_code=204)
    finally:
        db.close()


@app.get("/api/documentos/{doc_id}/download")
async def download_document(doc_id: str):
    """Descarga el archivo original de un documento."""
    db = SessionLocal()
    try:
        doc = db.query(Documento).filter(Documento.id == doc_id).first()
        if not doc or not doc.ruta_archivo_original:
            raise HTTPException(status_code=404, detail="Documento no encontrado")
        file_path = Path(doc.ruta_archivo_original)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Archivo no encontrado")
        filename = file_path.name
        return FileResponse(path=str(file_path), filename=filename, media_type="application/octet-stream")
    finally:
        db.close()


# ============================================================
# Endpoints: Configuración de Agentes IA
# ============================================================

DEFAULT_AGENTS = [
    {
        "id": "ocr",
        "nombre": "Agente OCR",
        "descripcion": "Extrae texto de imágenes y PDFs usando Tesseract y EasyOCR",
        "icono": "🔍",
        "tipo": "ocr",
        "modelo": "tesseract",
        "contexto": "por_documento",
        "prompt": "",
        "parametros": {
            "psm": 6,
            "lang": "spa+eng",
            "dpi": 300,
            "enhance_contrast": True
        }
    },
    {
        "id": "vision",
        "nombre": "Agente Visual",
        "descripcion": "Analiza visualmente el documento con IA para extraer campos adicionales",
        "icono": "👁",
        "tipo": "vision",
        "modelo": "moondream",
        "contexto": "por_documento",
        "prompt": "Analyze this document image and extract all visible information including: document type, provider name, RUT/tax ID, date, invoice number, amounts (net, tax, total), currency. Return as JSON.",
        "parametros": {
            "timeout": 120,
            "temperature": 0.1,
            "max_tokens": 512
        }
    },
    {
        "id": "extractor",
        "nombre": "Agente Extractor",
        "descripcion": "Extrae campos estructurados del texto usando IA y expresiones regulares",
        "icono": "🧠",
        "tipo": "extractor",
        "modelo": "qwen3.5:0.8b",
        "contexto": "por_documento",
        "prompt": "Eres un experto contable latinoamericano. Analiza el siguiente texto extraído de un documento contable y extrae los campos en formato JSON.\n\nTEXTO DEL DOCUMENTO:\n{text}\n\nExtrae EXACTAMENTE estos campos (usa null si no encuentras el valor):\n{{\n  \"tipo_documento\": \"FACTURA|BOLETA|RECIBO|NOTA_CREDITO|OTRO\",\n  \"proveedor\": \"nombre del emisor/vendedor\",\n  \"rut_proveedor\": \"RUT o RUC del emisor (ej: 77194706-9)\",\n  \"fecha_emision\": \"YYYY-MM-DD\",\n  \"folio\": \"número de documento/folio\",\n  \"descripcion\": \"descripción del producto o servicio\",\n  \"monto_neto\": numero sin IVA (solo número, sin símbolos),\n  \"iva\": monto del IVA (solo número),\n  \"monto_total\": monto total (solo número),\n  \"moneda\": \"CLP|USD|EUR|PEN|MXN|ARS\"\n}}\n\nResponde SOLO con el JSON, sin explicaciones.",
        "parametros": {
            "timeout": 90,
            "temperature": 0.1,
            "max_tokens": 512
        }
    },
    {
        "id": "clasificador",
        "nombre": "Agente Clasificador",
        "descripcion": "Clasifica el gasto en categorías contables usando IA y palabras clave",
        "icono": "🏷",
        "tipo": "clasificador",
        "modelo": "qwen3.5:0.8b",
        "contexto": "por_documento",
        "prompt": "Eres un contador experto. Clasifica este gasto empresarial en la categoría contable más apropiada.\n\nINFORMACION DEL GASTO:\n- Proveedor: {proveedor}\n- Descripción: {descripcion}\n- Tipo documento: {tipo_documento}\n- Monto: {monto_total} {moneda}\n\nCATEGORÍAS DISPONIBLES:\n{categorias}\n\nResponde en JSON:\n{{\n  \"categoria\": \"nombre exacto de la categoría\",\n  \"confianza\": 0.0 a 1.0,\n  \"razon\": \"explicación breve de por qué esta categoría\"\n}}\n\nResponde SOLO con el JSON.",
        "parametros": {
            "timeout": 60,
            "temperature": 0.1,
            "max_tokens": 256
        }
    },
    {
        "id": "auditor",
        "nombre": "Agente Auditor",
        "descripcion": "Detecta duplicados, anomalías y valida la coherencia del documento",
        "icono": "🛡",
        "tipo": "auditor",
        "modelo": "reglas",
        "contexto": "por_documento",
        "prompt": "",
        "parametros": {
            "check_duplicates": True,
            "check_amounts": True,
            "check_dates": True,
            "similarity_threshold": 0.85
        }
    },
    {
        "id": "recomendador",
        "nombre": "Agente Recomendador",
        "descripcion": "Analiza el historial de gastos y genera recomendaciones de optimización",
        "icono": "💡",
        "tipo": "recomendador",
        "modelo": "qwen3.5:0.8b",
        "contexto": "historial_mensual",
        "prompt": "Eres un asesor financiero experto en PYMEs latinoamericanas. Analiza el siguiente resumen de gastos del mes y genera recomendaciones concretas de optimización.\n\nRESUMEN DE GASTOS:\n{resumen}\n\nGira de la empresa: {giro}\n\nGenera 3-5 recomendaciones priorizadas en JSON:\n[{{\"titulo\": \"...\", \"descripcion\": \"...\", \"ahorro_estimado\": \"...\", \"prioridad\": \"alta|media|baja\"}}]",
        "parametros": {
            "timeout": 120,
            "temperature": 0.3,
            "max_tokens": 1024,
            "historial_meses": 3,
            "compactar_despues_de": 1
        }
    }
]

_AGENTS_FILE = DATA_DIR / "agents_config.json"

def _load_agents():
    if _AGENTS_FILE.exists():
        try:
            return json.loads(_AGENTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_AGENTS

def _save_agents(agents):
    _AGENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _AGENTS_FILE.write_text(json.dumps(agents, indent=2, ensure_ascii=False), encoding="utf-8")

@app.get("/api/agents")
async def get_agents():
    """Retorna la configuración de todos los agentes."""
    return {"agents": _load_agents()}

@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: str, config: dict):
    """Actualiza la configuración de un agente."""
    from fastapi import Body
    agents = _load_agents()
    for i, a in enumerate(agents):
        if a["id"] == agent_id:
            agents[i].update(config)
            agents[i]["id"] = agent_id  # Asegurar que el ID no cambie
            _save_agents(agents)
            # Si el modelo cambió, verificar disponibilidad
            new_model = config.get("modelo", "")
            if new_model and new_model not in ("tesseract", "reglas", "moondream"):
                try:
                    import requests as _req
                    r = _req.get("http://localhost:11434/api/tags", timeout=3)
                    models = [m["name"] for m in r.json().get("models", [])]
                    if new_model not in models:
                        return {"ok": True, "warning": f"Modelo '{new_model}' no encontrado en Ollama. Descarga con: ollama pull {new_model}"}
                except Exception:
                    pass
            return {"ok": True, "agent": agents[i]}
    raise HTTPException(status_code=404, detail=f"Agente '{agent_id}' no encontrado")

@app.post("/api/agents/reset")
async def reset_agents():
    """Restaura la configuración por defecto de los agentes."""
    _save_agents(DEFAULT_AGENTS)
    return {"ok": True, "agents": DEFAULT_AGENTS}


# ============================================================
# Servir archivos estáticos de la UI
# ============================================================

# Montar UI con ruta absoluta
_APP_DIR = BASE_DIR / "app"
if _APP_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_APP_DIR), html=True), name="ui")
else:
    print(f"WARNING: Directorio UI no encontrado en {_APP_DIR}")


# Redirigir raíz a la UI
@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/ui")


if __name__ == "__main__":
    # log_level="info" para que Pinokio detecte el puerto en el output
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        # loop="asyncio" es compatible con Windows (evita problemas con uvloop)
        loop="asyncio"
    )
