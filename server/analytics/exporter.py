"""
Módulo de exportación para pyme-ledger-ai.
Genera reportes en CSV y PDF.
"""
import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from io import StringIO, BytesIO
from sqlalchemy.orm import Session
import pandas as pd

from models import Documento, Empresa

# Intentar importar reportlab para PDF
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


class ExportEngine:
    """Motor de exportación de reportes."""
    
    def __init__(self, db: Session, empresa_id: str, data_dir: str = "data"):
        """
        Inicializa el motor de exportación.
        
        Args:
            db: Sesión de base de datos
            empresa_id: ID de la empresa
            data_dir: Directorio para guardar exportaciones
        """
        self.db = db
        self.empresa_id = empresa_id
        self.data_dir = Path(data_dir) / "exports"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def export_to_csv(self, period_days: int = 30, filename: str = None) -> Dict:
        """
        Exporta documentos a CSV.
        
        Args:
            period_days: Período en días
            filename: Nombre del archivo (default: auto-generado)
            
        Returns:
            Dict con información de la exportación
        """
        from datetime import timedelta
        
        start_date = datetime.utcnow() - timedelta(days=period_days)
        
        docs = self.db.query(Documento).filter(
            Documento.empresa_id == self.empresa_id,
            Documento.fecha_creacion >= start_date
        ).order_by(Documento.fecha_emision.desc()).all()
        
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"documentos_{timestamp}.csv"
        
        file_path = self.data_dir / filename
        
        # Preparar datos
        rows = []
        for doc in docs:
            row = {
                "ID": doc.id,
                "Fecha": doc.fecha_emision.strftime("%Y-%m-%d") if doc.fecha_emision else "",
                "Tipo": doc.tipo_documento.value if doc.tipo_documento else "",
                "Proveedor": doc.proveedor or "",
                "RUT": doc.rut_proveedor or "",
                "Folio": doc.folio or "",
                "Monto Neto": f"{doc.monto_neto:.2f}" if doc.monto_neto else "0.00",
                "IVA": f"{doc.iva:.2f}" if doc.iva else "0.00",
                "Total": f"{doc.monto_total:.2f}" if doc.monto_total else "0.00",
                "Moneda": doc.moneda or "CLP",
                "Categoría": doc.categoria.nombre if doc.categoria else "Sin asignar",
                "Centro Costo": doc.centro_costo.nombre if doc.centro_costo else "Sin asignar",
                "Confianza": f"{doc.confianza_clasificacion:.0%}" if doc.confianza_clasificacion else "0%",
                "Estado": doc.estado_revision.value if doc.estado_revision else "",
                "Excepciones": self._format_exceptions(doc.excepciones)
            }
            rows.append(row)
        
        # Escribir CSV
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
        else:
            # Crear CSV vacío con headers
            with open(file_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "ID", "Fecha", "Tipo", "Proveedor", "RUT", "Folio",
                    "Monto Neto", "IVA", "Total", "Moneda", "Categoría",
                    "Centro Costo", "Confianza", "Estado", "Excepciones"
                ])
                writer.writeheader()
        
        return {
            "tipo": "CSV",
            "archivo": str(file_path),
            "filename": filename,
            "documentos": len(docs),
            "tamaño_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "fecha_generacion": datetime.utcnow().isoformat()
        }
    
    def export_to_pdf(self, analytics_data: Dict, filename: str = None) -> Dict:
        """
        Exporta reporte a PDF.
        
        Args:
            analytics_data: Datos de analítica para incluir
            filename: Nombre del archivo
            
        Returns:
            Dict con información de la exportación
        """
        if not HAS_REPORTLAB:
            return {
                "error": "reportlab no está instalado",
                "sugerencia": "pip install reportlab"
            }
        
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"reporte_{timestamp}.pdf"
        
        file_path = self.data_dir / filename
        
        try:
            # Obtener empresa
            empresa = self.db.query(Empresa).filter(
                Empresa.id == self.empresa_id
            ).first()
            
            # Crear documento PDF
            doc = SimpleDocTemplate(str(file_path), pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            # Estilo personalizado
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1e293b'),
                spaceAfter=30,
                alignment=TA_CENTER
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#334155'),
                spaceAfter=12,
                spaceBefore=12
            )
            
            # Título
            story.append(Paragraph(f"Reporte de Gastos - {empresa.nombre_fantasia or empresa.razon_social}", title_style))
            story.append(Spacer(1, 0.3*inch))
            
            # Información general
            story.append(Paragraph("Información General", heading_style))
            
            general_data = [
                ["Empresa:", empresa.razon_social],
                ["RUT:", empresa.rut],
                ["Período:", f"{analytics_data.get('periodo_dias', 30)} días"],
                ["Fecha Generación:", datetime.utcnow().strftime("%d/%m/%Y %H:%M")]
            ]
            
            general_table = Table(general_data, colWidths=[2*inch, 4*inch])
            general_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e2e8f0')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(general_table)
            story.append(Spacer(1, 0.3*inch))
            
            # KPIs
            if analytics_data.get('kpis'):
                story.append(Paragraph("Indicadores Clave (KPIs)", heading_style))
                
                kpis = analytics_data['kpis']
                kpi_data = [
                    ["Métrica", "Valor"],
                    ["Total Gasto", f"${kpis.get('total_gasto', 0):,.2f}"],
                    ["Total Neto", f"${kpis.get('total_neto', 0):,.2f}"],
                    ["Total IVA", f"${kpis.get('total_iva', 0):,.2f}"],
                    ["Promedio por Documento", f"${kpis.get('promedio_documento', 0):,.2f}"],
                    ["Cantidad de Documentos", str(kpis.get('cantidad_documentos', 0))],
                    ["Pendiente Revisión", str(kpis.get('documentos_pendiente_revision', 0))],
                    ["Revisados", str(kpis.get('documentos_revisados', 0))]
                ]
                
                kpi_table = Table(kpi_data, colWidths=[3*inch, 3*inch])
                kpi_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f1f5f9')])
                ]))
                story.append(kpi_table)
                story.append(Spacer(1, 0.3*inch))
            
            # Categorías
            if analytics_data.get('categorias'):
                story.append(Paragraph("Gastos por Categoría", heading_style))
                
                cat_data = [["Categoría", "Total", "Porcentaje", "Cantidad"]]
                for cat in analytics_data['categorias'][:10]:  # Top 10
                    cat_data.append([
                        cat.get('categoria', ''),
                        f"${cat.get('total', 0):,.2f}",
                        f"{cat.get('porcentaje', 0):.1f}%",
                        str(cat.get('cantidad', 0))
                    ])
                
                cat_table = Table(cat_data, colWidths=[2.5*inch, 1.5*inch, 1*inch, 1*inch])
                cat_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f1f5f9')])
                ]))
                story.append(cat_table)
            
            # Generar PDF
            doc.build(story)
            
            return {
                "tipo": "PDF",
                "archivo": str(file_path),
                "filename": filename,
                "tamaño_bytes": file_path.stat().st_size if file_path.exists() else 0,
                "fecha_generacion": datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            return {
                "error": f"Error generando PDF: {str(e)}"
            }
    
    def export_accounting_ledger(self, period_days: int = 30, filename: str = None) -> Dict:
        """
        Exporta libro contable en formato estándar.
        
        Args:
            period_days: Período en días
            filename: Nombre del archivo
            
        Returns:
            Dict con información de la exportación
        """
        from datetime import timedelta
        
        start_date = datetime.utcnow() - timedelta(days=period_days)
        
        docs = self.db.query(Documento).filter(
            Documento.empresa_id == self.empresa_id,
            Documento.fecha_creacion >= start_date
        ).order_by(Documento.fecha_emision).all()
        
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"libro_contable_{timestamp}.csv"
        
        file_path = self.data_dir / filename
        
        # Formato de libro contable
        rows = []
        rows.append({
            "Fecha": "Fecha",
            "Descripción": "Descripción",
            "Cuenta": "Cuenta",
            "Debe": "Debe",
            "Haber": "Haber",
            "Saldo": "Saldo"
        })
        
        saldo = 0.0
        for doc in docs:
            # Entrada de gasto
            monto = doc.monto_total or 0
            saldo -= monto
            
            rows.append({
                "Fecha": doc.fecha_emision.strftime("%Y-%m-%d") if doc.fecha_emision else "",
                "Descripción": f"{doc.tipo_documento.value if doc.tipo_documento else 'Documento'} - {doc.proveedor or 'Desconocido'}",
                "Cuenta": doc.categoria.codigo if doc.categoria else "0000",
                "Debe": f"{monto:.2f}",
                "Haber": "0.00",
                "Saldo": f"{saldo:.2f}"
            })
        
        # Escribir CSV
        if rows:
            df = pd.DataFrame(rows[1:])  # Excluir header manual
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
        
        return {
            "tipo": "Libro Contable CSV",
            "archivo": str(file_path),
            "filename": filename,
            "documentos": len(docs),
            "saldo_final": f"{saldo:.2f}",
            "tamaño_bytes": file_path.stat().st_size if file_path.exists() else 0,
            "fecha_generacion": datetime.utcnow().isoformat()
        }
    
    def _format_exceptions(self, exceptions_json: str) -> str:
        """Formatea excepciones para CSV."""
        if not exceptions_json:
            return ""
        
        try:
            exceptions = json.loads(exceptions_json)
            if isinstance(exceptions, list):
                return "; ".join([e.get("message", "") for e in exceptions[:3]])
            return ""
        except:
            return ""


# Instancia global
export_engine = None


def get_export_engine(db: Session, empresa_id: str, data_dir: str = "data") -> ExportEngine:
    """Obtiene una instancia del motor de exportación."""
    return ExportEngine(db, empresa_id, data_dir)
