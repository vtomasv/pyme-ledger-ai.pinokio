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
from fastapi.responses import RedirectResponse, FileResponse
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
from analytics import AnalyticsEngine
from analytics.recommender import get_recommender_engine
from analytics.exporter import get_export_engine

# Configuración
PORT = int(os.environ.get("PORT", 8000))
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
PLUGIN_DIR = Path(os.environ.get("PLUGIN_DIR", "."))
UPLOADS_DIR = DATA_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

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
    razon_social: str
    nombre_fantasia: Optional[str] = None
    rut: str
    pais: str = "Chile"
    moneda_base: str = "CLP"
    regimen_tributario: Optional[str] = None


class CentroCostoCreate(BaseModel):
    codigo: str
    nombre: str
    descripcion: Optional[str] = None


class CategoriaContableCreate(BaseModel):
    codigo: str
    nombre: str
    tipo_gasto: str
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
        new_empresa = Empresa(
            id=empresa_id,
            razon_social=empresa.razon_social,
            nombre_fantasia=empresa.nombre_fantasia,
            rut=empresa.rut,
            pais=empresa.pais,
            moneda_base=empresa.moneda_base,
            regimen_tributario=empresa.regimen_tributario
        )
        db.add(new_empresa)
        db.commit()
        
        return {
            "id": empresa_id,
            "razon_social": empresa.razon_social,
            "rut": empresa.rut,
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
            codigo=cat.codigo,
            nombre=cat.nombre,
            tipo_gasto=cat.tipo_gasto,
            deducibilidad=cat.deducibilidad,
            regla_iva=cat.regla_iva
        )
        db.add(new_cat)
        db.commit()
        
        return {"id": cat_id, "codigo": cat.codigo, "nombre": cat.nombre}
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
    """Carga y procesa un documento."""
    db = SessionLocal()
    try:
        # Verificar empresa
        empresa = db.query(Empresa).filter(Empresa.id == empresa_id).first()
        if not empresa:
            raise HTTPException(status_code=404, detail="Empresa no encontrada")
        
        # Guardar archivo
        file_extension = Path(file.filename).suffix
        file_id = str(uuid.uuid4())
        file_path = UPLOADS_DIR / f"{file_id}{file_extension}"
        
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Procesar documento
        pipeline = DocumentProcessingPipeline(db, empresa_id)
        result = pipeline.process_document(str(file_path))
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


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
                    "fecha": d.fecha_emision.isoformat() if d.fecha_emision else None,
                    "proveedor": d.proveedor,
                    "monto_total": d.monto_total,
                    "categoria": d.categoria.nombre if d.categoria else None,
                    "estado": d.estado_revision.value if d.estado_revision else None,
                    "confianza": d.confianza_clasificacion
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
        
        return {
            "id": doc.id,
            "fecha": doc.fecha_emision.isoformat() if doc.fecha_emision else None,
            "tipo": doc.tipo_documento.value if doc.tipo_documento else None,
            "proveedor": doc.proveedor,
            "rut": doc.rut_proveedor,
            "folio": doc.folio,
            "monto_neto": doc.monto_neto,
            "iva": doc.iva,
            "monto_total": doc.monto_total,
            "moneda": doc.moneda,
            "categoria": doc.categoria.nombre if doc.categoria else None,
            "centro_costo": doc.centro_costo.nombre if doc.centro_costo else None,
            "confianza": doc.confianza_clasificacion,
            "estado": doc.estado_revision.value if doc.estado_revision else None,
            "excepciones": excepciones,
            "texto_extraido": doc.texto_extraido[:500] if doc.texto_extraido else None
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
# Servir archivos estáticos de la UI
# ============================================================

app.mount("/ui", StaticFiles(directory=str(PLUGIN_DIR / "app"), html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
