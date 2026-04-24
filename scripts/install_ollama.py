"""
install_ollama.py — Instala y arranca Ollama de forma cross-platform.
Ejecutado por install.json. Funciona en Windows, macOS y Linux.

En Windows:
  1. Intenta winget install Ollama.Ollama
  2. Si winget falla, descarga el instalador .exe directamente
  3. Agrega Ollama al PATH de la sesión actual
  4. Arranca ollama serve como proceso de fondo

En macOS/Linux:
  1. Verifica si ollama ya está instalado
  2. Si no, usa curl | sh (Linux) o brew (macOS)
  3. Arranca ollama serve como proceso de fondo
"""

import os
import sys
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

OLLAMA_URL = "http://127.0.0.1:11434"


def log(msg: str):
    print(f"[install_ollama] {msg}", flush=True)


def is_ollama_installed() -> bool:
    """Verifica si ollama está disponible en el PATH."""
    return shutil.which("ollama") is not None


def is_ollama_running() -> bool:
    """Verifica si Ollama está respondiendo en el puerto."""
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_ollama_windows() -> str:
    """Busca el ejecutable de Ollama en rutas comunes de Windows."""
    common_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Ollama" / "ollama.exe",
        Path(os.environ.get("USERPROFILE", "")) / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        Path("C:/Users") / os.environ.get("USERNAME", "user") / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
    ]
    for p in common_paths:
        if p.exists():
            return str(p)
    return ""


def add_ollama_to_path_windows():
    """Agrega el directorio de Ollama al PATH de la sesión actual."""
    ollama_exe = find_ollama_windows()
    if ollama_exe:
        ollama_dir = str(Path(ollama_exe).parent)
        current_path = os.environ.get("PATH", "")
        if ollama_dir.lower() not in current_path.lower():
            os.environ["PATH"] = ollama_dir + os.pathsep + current_path
            log(f"Ollama agregado al PATH: {ollama_dir}")
        return True
    return False


def install_ollama_windows():
    """Instala Ollama en Windows."""
    # Intentar winget primero
    if shutil.which("winget"):
        log("Intentando instalar Ollama via winget...")
        try:
            result = subprocess.run(
                ["winget", "install", "-e", "--id", "Ollama.Ollama",
                 "--accept-source-agreements", "--accept-package-agreements",
                 "--silent"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0 or "already installed" in result.stdout.lower():
                log("Ollama instalado via winget.")
                add_ollama_to_path_windows()
                return True
        except Exception as e:
            log(f"winget falló: {e}")

    # Fallback: descargar el instalador directamente
    log("Descargando instalador de Ollama...")
    installer_url = "https://ollama.com/download/OllamaSetup.exe"
    installer_path = Path(os.environ.get("TEMP", "/tmp")) / "OllamaSetup.exe"

    try:
        urllib.request.urlretrieve(installer_url, str(installer_path))
        log(f"Instalador descargado en {installer_path}")

        # Ejecutar instalador en modo silencioso
        log("Ejecutando instalador silencioso...")
        result = subprocess.run(
            [str(installer_path), "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            log("Ollama instalado correctamente.")
            add_ollama_to_path_windows()
            return True
        else:
            log(f"Instalador retornó código {result.returncode}")
    except Exception as e:
        log(f"Error descargando/instalando: {e}")

    # Último intento: buscar en rutas comunes
    if add_ollama_to_path_windows():
        log("Ollama encontrado en rutas comunes.")
        return True

    log("ERROR: No se pudo instalar Ollama automáticamente.")
    log("Por favor instala Ollama manualmente desde https://ollama.com/download")
    return False


def install_ollama_unix():
    """Instala Ollama en macOS/Linux."""
    if IS_MACOS and shutil.which("brew"):
        log("Instalando Ollama via Homebrew...")
        try:
            subprocess.run(["brew", "install", "ollama"],
                           capture_output=True, timeout=300)
            if shutil.which("ollama"):
                return True
        except Exception:
            pass

    if IS_LINUX:
        log("Instalando Ollama via script oficial...")
        try:
            subprocess.run(
                ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                timeout=300
            )
            if shutil.which("ollama"):
                return True
        except Exception as e:
            log(f"Error: {e}")

    log("ERROR: No se pudo instalar Ollama.")
    log("Instala manualmente desde https://ollama.com/download")
    return False


def start_ollama():
    """Arranca ollama serve como proceso de fondo."""
    if is_ollama_running():
        log("Ollama ya está corriendo.")
        return True

    ollama_cmd = shutil.which("ollama")
    if not ollama_cmd:
        # En Windows, buscar en rutas comunes
        if IS_WINDOWS:
            ollama_cmd = find_ollama_windows()
        if not ollama_cmd:
            log("No se encontró el ejecutable de ollama.")
            return False

    log("Iniciando ollama serve...")
    try:
        if IS_WINDOWS:
            # En Windows, usar CREATE_NO_WINDOW para no bloquear
            CREATE_NO_WINDOW = 0x08000000
            subprocess.Popen(
                [ollama_cmd, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW
            )
        else:
            subprocess.Popen(
                [ollama_cmd, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
    except Exception as e:
        log(f"Error iniciando ollama serve: {e}")
        return False

    # Esperar a que esté listo
    for i in range(15):
        time.sleep(2)
        if is_ollama_running():
            log("Ollama está listo.")
            return True
        log(f"Esperando a Ollama... ({i+1}/15)")

    log("AVISO: Ollama no respondió después de 30 segundos.")
    return False


def main():
    log(f"Plataforma: {sys.platform}")

    # 1. Verificar si ya está instalado
    if is_ollama_installed():
        log("Ollama ya está instalado.")
    elif IS_WINDOWS:
        # Buscar en rutas comunes primero
        if add_ollama_to_path_windows() and is_ollama_installed():
            log("Ollama encontrado en rutas comunes de Windows.")
        else:
            install_ollama_windows()
    else:
        install_ollama_unix()

    # 2. Arrancar Ollama
    started = start_ollama()

    if not started:
        log("AVISO: Ollama no está corriendo. Los modelos se descargarán cuando esté disponible.")
    else:
        log("Ollama instalado y corriendo correctamente.")


if __name__ == "__main__":
    main()
