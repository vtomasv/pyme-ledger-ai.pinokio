"""
setup.py — Script de instalación cross-platform para pyme-ledger-ai.
Ejecutado por install.json. Funciona en macOS, Windows y Linux.

Responsabilidades:
  1. Crear el venv si no existe
  2. Instalar dependencias con el pip del venv (ruta explícita)
  3. Crear estructura de directorios de datos
  4. Copiar defaults si no existen
  5. Verificar e instalar dependencias del sistema (Tesseract, Poppler)
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path

# ── Rutas absolutas ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
VENV_DIR = BASE_DIR / "venv"
DEFAULTS_DIR = BASE_DIR / "defaults"
DATA_DIR = BASE_DIR / "data"

# Rutas del venv según plataforma
IS_WINDOWS = sys.platform == "win32"
VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
VENV_PIP    = VENV_DIR / ("Scripts/pip.exe"    if IS_WINDOWS else "bin/pip")

def log(msg: str):
    print(f"[setup] {msg}", flush=True)

def run(cmd: list, **kwargs):
    """Ejecuta un comando y lanza excepción si falla."""
    log(f"Ejecutando: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        raise RuntimeError(f"Comando falló con código {result.returncode}: {cmd}")
    return result

# ── 1. Crear venv ──────────────────────────────────────────────────────────────
def create_venv():
    if VENV_PYTHON.exists():
        log(f"Venv ya existe en {VENV_DIR}")
        return
    log(f"Creando venv en {VENV_DIR} ...")
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    log("Venv creado.")

# ── 2. Instalar dependencias Python ───────────────────────────────────────────
def install_python_deps():
    req_file = BASE_DIR / "requirements.txt"
    log("Actualizando pip ...")
    run([str(VENV_PIP), "install", "--upgrade", "pip"])
    log("Instalando requirements.txt ...")
    run([str(VENV_PIP), "install", "-r", str(req_file)])
    log("Dependencias Python instaladas.")

# ── 3. Crear estructura de datos ──────────────────────────────────────────────
def create_data_dirs():
    dirs = [
        DATA_DIR / "agents",
        DATA_DIR / "prompts" / "system",
        DATA_DIR / "prompts" / "templates",
        DATA_DIR / "sessions",
        DATA_DIR / "exports",
        DATA_DIR / "uploads",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    log("Directorios de datos creados.")

# ── 4. Copiar defaults ────────────────────────────────────────────────────────
def copy_defaults():
    agents_src = DEFAULTS_DIR / "agents.json"
    agents_dst = DATA_DIR / "agents" / "agents.json"
    if agents_src.exists() and not agents_dst.exists():
        shutil.copy2(str(agents_src), str(agents_dst))
        log("agents.json copiado.")

    prompts_src = DEFAULTS_DIR / "prompts"
    prompts_dst = DATA_DIR / "prompts"
    if prompts_src.exists():
        for src_file in prompts_src.rglob("*"):
            if src_file.is_file():
                rel = src_file.relative_to(prompts_src)
                dst_file = prompts_dst / rel
                if not dst_file.exists():
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(src_file), str(dst_file))
        log("Prompts por defecto copiados.")

# ── 5. Verificar dependencias del sistema ─────────────────────────────────────
def check_system_deps():
    """
    Tesseract y Poppler son binarios del sistema, no paquetes pip.
    Detectamos si están disponibles y mostramos instrucciones claras si no.
    El plugin funciona sin ellos (OCR fallback a texto directo de PDF).
    """
    warnings = []

    # Tesseract
    if not shutil.which("tesseract"):
        warnings.append(
            "AVISO: Tesseract OCR no encontrado. "
            "El OCR de imágenes escaneadas usará modo básico. "
            "Para activarlo: macOS→'brew install tesseract', "
            "Windows→descargar desde github.com/UB-Mannheim/tesseract/wiki, "
            "Linux→'sudo apt install tesseract-ocr'"
        )

    # Poppler (para pdf2image)
    if not shutil.which("pdftoppm") and not shutil.which("pdfinfo"):
        warnings.append(
            "AVISO: Poppler no encontrado. "
            "La conversión de PDF a imagen usará modo básico (solo texto). "
            "Para activarlo: macOS→'brew install poppler', "
            "Windows→descargar desde github.com/oschwartz10612/poppler-windows, "
            "Linux→'sudo apt install poppler-utils'"
        )

    for w in warnings:
        log(w)

    if not warnings:
        log("Dependencias del sistema: OK (Tesseract y Poppler encontrados).")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log(f"=== Iniciando setup en {sys.platform} ({BASE_DIR}) ===")
    try:
        create_venv()
        install_python_deps()
        create_data_dirs()
        copy_defaults()
        check_system_deps()
        log("=== Setup completado exitosamente ===")
        sys.exit(0)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)
