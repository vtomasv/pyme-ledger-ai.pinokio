"""
setup.py — Script de instalación cross-platform para pyme-ledger-ai.
Ejecutado por install.json. Funciona en macOS, Windows y Linux.

Responsabilidades:
  1. Crear el venv si no existe
  2. Instalar dependencias Python con el pip del venv (ruta explícita)
  3. Instalar Tesseract OCR automáticamente según plataforma
  4. Instalar Poppler (para PDF→imagen) automáticamente según plataforma
  5. Crear estructura de directorios de datos
  6. Copiar defaults si no existen
  7. Descargar modelo de visión de Ollama para OCR avanzado
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ── Rutas absolutas ────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
VENV_DIR    = BASE_DIR / "venv"
DEFAULTS_DIR = BASE_DIR / "defaults"
DATA_DIR    = BASE_DIR / "data"

IS_WINDOWS  = sys.platform == "win32"
IS_MACOS    = sys.platform == "darwin"
IS_LINUX    = sys.platform.startswith("linux")

VENV_PYTHON = VENV_DIR / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")
VENV_PIP    = VENV_DIR / ("Scripts/pip.exe"    if IS_WINDOWS else "bin/pip")


def log(msg: str):
    print(f"[setup] {msg}", flush=True)


def run(cmd: list, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Ejecuta un comando. Si check=True lanza excepción si falla."""
    log(f"$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        raise RuntimeError(f"Comando falló (código {result.returncode}): {cmd}")
    return result


def run_silent(cmd: list) -> bool:
    """Ejecuta un comando silenciosamente. Retorna True si tuvo éxito."""
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300)
        return r.returncode == 0
    except Exception:
        return False


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


# ── 3. Instalar Tesseract OCR ─────────────────────────────────────────────────
def install_tesseract():
    """
    Instala Tesseract OCR automáticamente según la plataforma.
    Tesseract es necesario para OCR de imágenes escaneadas.
    """
    if shutil.which("tesseract"):
        log("Tesseract ya está instalado.")
        return True

    log("Instalando Tesseract OCR...")

    if IS_MACOS:
        # Homebrew
        if shutil.which("brew"):
            ok = run_silent(["brew", "install", "tesseract"])
            if ok:
                log("Tesseract instalado via Homebrew.")
                return True
        # MacPorts
        if shutil.which("port"):
            ok = run_silent(["sudo", "port", "install", "tesseract"])
            if ok:
                log("Tesseract instalado via MacPorts.")
                return True
        log("AVISO: No se pudo instalar Tesseract automáticamente en macOS.")
        log("  Instala manualmente: brew install tesseract")
        return False

    elif IS_LINUX:
        # Detectar gestor de paquetes
        if shutil.which("apt-get"):
            ok = run_silent(["sudo", "apt-get", "install", "-y",
                             "tesseract-ocr", "tesseract-ocr-spa",
                             "tesseract-ocr-eng"])
            if ok:
                log("Tesseract instalado via apt-get.")
                return True
        elif shutil.which("dnf"):
            ok = run_silent(["sudo", "dnf", "install", "-y",
                             "tesseract", "tesseract-langpack-spa"])
            if ok:
                log("Tesseract instalado via dnf.")
                return True
        elif shutil.which("pacman"):
            ok = run_silent(["sudo", "pacman", "-S", "--noconfirm",
                             "tesseract", "tesseract-data-spa"])
            if ok:
                log("Tesseract instalado via pacman.")
                return True
        log("AVISO: No se pudo instalar Tesseract automáticamente en Linux.")
        log("  Instala manualmente: sudo apt-get install tesseract-ocr tesseract-ocr-spa")
        return False

    elif IS_WINDOWS:
        # En Windows, intentar via winget o chocolatey
        if shutil.which("winget"):
            ok = run_silent(["winget", "install", "-e", "--id",
                             "UB-Mannheim.TesseractOCR",
                             "--accept-source-agreements",
                             "--accept-package-agreements"])
            if ok:
                log("Tesseract instalado via winget.")
                return True
        if shutil.which("choco"):
            ok = run_silent(["choco", "install", "tesseract", "-y"])
            if ok:
                log("Tesseract instalado via Chocolatey.")
                return True
        log("AVISO: No se pudo instalar Tesseract automáticamente en Windows.")
        log("  Descarga el instalador desde: https://github.com/UB-Mannheim/tesseract/wiki")
        return False

    return False


# ── 4. Instalar Poppler ───────────────────────────────────────────────────────
def install_poppler():
    """
    Instala Poppler (necesario para convertir PDFs a imágenes).
    """
    if shutil.which("pdftoppm") or shutil.which("pdfinfo"):
        log("Poppler ya está instalado.")
        return True

    log("Instalando Poppler (para procesamiento de PDFs)...")

    if IS_MACOS:
        if shutil.which("brew"):
            ok = run_silent(["brew", "install", "poppler"])
            if ok:
                log("Poppler instalado via Homebrew.")
                return True
        log("AVISO: Instala Poppler manualmente: brew install poppler")
        return False

    elif IS_LINUX:
        if shutil.which("apt-get"):
            ok = run_silent(["sudo", "apt-get", "install", "-y", "poppler-utils"])
            if ok:
                log("Poppler instalado via apt-get.")
                return True
        elif shutil.which("dnf"):
            ok = run_silent(["sudo", "dnf", "install", "-y", "poppler-utils"])
            if ok:
                log("Poppler instalado via dnf.")
                return True
        log("AVISO: Instala Poppler manualmente: sudo apt-get install poppler-utils")
        return False

    elif IS_WINDOWS:
        log("AVISO: En Windows, descarga Poppler desde:")
        log("  https://github.com/oschwartz10612/poppler-windows/releases")
        return False

    return False


# ── 5. Crear estructura de datos ──────────────────────────────────────────────
def create_data_dirs():
    dirs = [
        DATA_DIR / "uploads",
        DATA_DIR / "exports",
        DATA_DIR / "sessions",
        DATA_DIR / "agents",
        DATA_DIR / "prompts" / "system",
        DATA_DIR / "prompts" / "templates",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    log("Directorios de datos creados.")


# ── 6. Copiar defaults ────────────────────────────────────────────────────────
def copy_defaults():
    if not DEFAULTS_DIR.exists():
        return

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


# ── 7. Descargar modelo de visión de Ollama ───────────────────────────────────
def pull_vision_model():
    """
    Descarga el modelo de visión más ligero disponible para OCR avanzado.
    moondream (1.8B) es el más ligero y funciona bien para documentos.
    """
    if not shutil.which("ollama"):
        log("AVISO: Ollama no encontrado. El OCR avanzado con VLLM no estará disponible.")
        return False

    # Verificar si ya hay un modelo de visión
    try:
        result = subprocess.run(["ollama", "list"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            output = result.stdout.lower()
            # Si ya tiene llama3.2-vision, llava o moondream, no descargar
            vision_models = ["moondream", "llava", "llama3.2-vision",
                             "granite3.2-vision", "minicpm-v"]
            for vm in vision_models:
                if vm in output:
                    log(f"Modelo de visión ya disponible: {vm}")
                    return True
    except Exception:
        pass

    # Descargar moondream (el más ligero: ~1.8GB)
    log("Descargando modelo de visión 'moondream' (1.8GB) para OCR avanzado...")
    log("Este modelo permite leer documentos manuscritos y complejos.")
    ok = run_silent(["ollama", "pull", "moondream"])
    if ok:
        log("Modelo moondream descargado exitosamente.")
        return True
    else:
        log("AVISO: No se pudo descargar el modelo de visión.")
        log("  El OCR funcionará con Tesseract/EasyOCR para documentos estándar.")
        return False


# ── Resumen de instalación ────────────────────────────────────────────────────
def write_install_summary(results: dict):
    """Guarda un resumen de la instalación para diagnóstico."""
    summary = {
        "platform": sys.platform,
        "python": sys.version,
        "arch": platform.machine(),
        "installed_at": __import__('datetime').datetime.now().isoformat(),
        "components": results
    }
    summary_path = DATA_DIR / "install_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"Resumen guardado en {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log(f"=== Iniciando setup en {sys.platform} / {platform.machine()} ===")
    log(f"=== Directorio base: {BASE_DIR} ===")

    results = {}
    errors = []

    try:
        log("\n--- Paso 1/7: Creando entorno virtual Python ---")
        create_venv()
        results['venv'] = 'ok'
    except Exception as e:
        log(f"ERROR en venv: {e}")
        errors.append(f"venv: {e}")
        sys.exit(1)  # Sin venv no podemos continuar

    try:
        log("\n--- Paso 2/7: Instalando dependencias Python ---")
        install_python_deps()
        results['python_deps'] = 'ok'
    except Exception as e:
        log(f"ERROR en dependencias Python: {e}")
        errors.append(f"python_deps: {e}")
        sys.exit(1)  # Sin dependencias no podemos continuar

    try:
        log("\n--- Paso 3/7: Instalando Tesseract OCR ---")
        ok = install_tesseract()
        results['tesseract'] = 'ok' if ok else 'warning_not_installed'
    except Exception as e:
        log(f"AVISO en Tesseract: {e}")
        results['tesseract'] = f'error: {e}'

    try:
        log("\n--- Paso 4/7: Instalando Poppler (PDF→imagen) ---")
        ok = install_poppler()
        results['poppler'] = 'ok' if ok else 'warning_not_installed'
    except Exception as e:
        log(f"AVISO en Poppler: {e}")
        results['poppler'] = f'error: {e}'

    try:
        log("\n--- Paso 5/7: Creando directorios de datos ---")
        create_data_dirs()
        results['data_dirs'] = 'ok'
    except Exception as e:
        log(f"ERROR en directorios: {e}")
        errors.append(f"data_dirs: {e}")

    try:
        log("\n--- Paso 6/7: Copiando configuración por defecto ---")
        copy_defaults()
        results['defaults'] = 'ok'
    except Exception as e:
        log(f"AVISO en defaults: {e}")
        results['defaults'] = f'warning: {e}'

    try:
        log("\n--- Paso 7/7: Descargando modelo de visión para OCR avanzado ---")
        ok = pull_vision_model()
        results['vision_model'] = 'ok' if ok else 'warning_not_available'
    except Exception as e:
        log(f"AVISO en modelo de visión: {e}")
        results['vision_model'] = f'warning: {e}'

    # Resumen final
    write_install_summary(results)

    if errors:
        log(f"\n=== Setup completado con errores: {errors} ===")
        sys.exit(1)
    else:
        log("\n=== Setup completado exitosamente ===")
        log(f"Componentes: {results}")
        sys.exit(0)
