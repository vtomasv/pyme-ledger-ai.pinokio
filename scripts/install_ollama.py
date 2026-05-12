"""
install_ollama.py — Instala y arranca Ollama de forma cross-platform.
Ejecutado por install.json. Funciona en Windows, macOS y Linux.

COMPATIBILIDAD WINDOWS:
  - Busca Ollama en rutas comunes de Windows (no depende del PATH)
  - Usa CREATE_NO_WINDOW para procesos de fondo
  - Usa urllib (stdlib) en lugar de curl para health checks
  - No usa /dev/null, &, ni otros comandos Unix
  - Maneja correctamente rutas con espacios (Program Files)

COMPATIBILIDAD macOS:
  - Verifica si Ollama.app está instalado como aplicación
  - Usa brew como fallback
  - Arranca via `open -a Ollama` si está como .app
"""

import os
import sys
import shutil
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")

OLLAMA_URL = "http://127.0.0.1:11434"


def log(msg: str):
    print(f"[install_ollama] {msg}", flush=True)


def is_ollama_running() -> bool:
    """Verifica si Ollama está respondiendo en el puerto."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_ollama_executable() -> str:
    """Busca el ejecutable de Ollama en el sistema."""
    # 1. Buscar en PATH
    ollama = shutil.which("ollama")
    if ollama:
        return ollama

    # 2. En Windows, buscar en rutas comunes
    if IS_WINDOWS:
        username = os.environ.get("USERNAME", "")
        common_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Ollama" / "ollama.exe",
            Path(os.environ.get("USERPROFILE", "")) / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
            Path(r"C:\Users") / username / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
            # Ollama también puede estar en el directorio del usuario
            Path(os.environ.get("USERPROFILE", "")) / "ollama.exe",
        ]
        for p in common_paths:
            try:
                if p.exists():
                    # Agregar al PATH de la sesión
                    ollama_dir = str(p.parent)
                    os.environ["PATH"] = ollama_dir + os.pathsep + os.environ.get("PATH", "")
                    return str(p)
            except (OSError, ValueError):
                continue

    # 3. En macOS, verificar si está como aplicación
    if IS_MACOS:
        app_ollama = Path("/Applications/Ollama.app/Contents/Resources/ollama")
        if app_ollama.exists():
            return str(app_ollama)
        # También verificar en /usr/local/bin
        usr_local = Path("/usr/local/bin/ollama")
        if usr_local.exists():
            return str(usr_local)

    return ""


def install_ollama_windows() -> bool:
    """Instala Ollama en Windows."""
    # Intentar winget primero (más confiable)
    if shutil.which("winget"):
        log("Intentando instalar Ollama via winget...")
        try:
            result = subprocess.run(
                ["winget", "install", "-e", "--id", "Ollama.Ollama",
                 "--accept-source-agreements", "--accept-package-agreements",
                 "--silent"],
                capture_output=True, text=True, timeout=600
            )
            stdout_lower = (result.stdout or "").lower()
            if result.returncode == 0 or "already installed" in stdout_lower or "ya está instalado" in stdout_lower:
                log("Ollama instalado/encontrado via winget.")
                # Esperar un momento para que se registre en el sistema
                time.sleep(3)
                if find_ollama_executable():
                    return True
        except subprocess.TimeoutExpired:
            log("winget tardó demasiado. Intentando descarga directa...")
        except Exception as e:
            log(f"winget falló: {e}")

    # Fallback: descargar el instalador directamente
    log("Descargando instalador de Ollama desde ollama.com...")
    installer_url = "https://ollama.com/download/OllamaSetup.exe"
    temp_dir = Path(os.environ.get("TEMP", os.environ.get("TMP", ".")))
    installer_path = temp_dir / "OllamaSetup.exe"

    try:
        urllib.request.urlretrieve(installer_url, str(installer_path))
        log(f"Instalador descargado ({installer_path})")

        # Ejecutar instalador en modo silencioso
        log("Ejecutando instalador silencioso...")
        result = subprocess.run(
            [str(installer_path), "/VERYSILENT", "/NORESTART", "/SUPPRESSMSGBOXES"],
            capture_output=True, text=True, timeout=300
        )
        # Esperar a que se registre
        time.sleep(5)

        if find_ollama_executable():
            log("Ollama instalado correctamente.")
            return True
        else:
            log(f"Instalador terminó (código {result.returncode}) pero ollama no encontrado.")
    except Exception as e:
        log(f"Error descargando/instalando: {e}")
    finally:
        # Limpiar instalador
        try:
            installer_path.unlink(missing_ok=True)
        except Exception:
            pass

    return False


def install_ollama_macos() -> bool:
    """Instala Ollama en macOS."""
    # Verificar si Ollama.app ya está instalado
    if Path("/Applications/Ollama.app").exists():
        log("Ollama.app ya está instalado.")
        return True

    # Intentar brew
    if shutil.which("brew"):
        log("Instalando Ollama via Homebrew...")
        try:
            result = subprocess.run(
                ["brew", "install", "ollama"],
                capture_output=True, text=True, timeout=300
            )
            if result.returncode == 0 or find_ollama_executable():
                log("Ollama instalado via Homebrew.")
                return True
        except Exception as e:
            log(f"Homebrew falló: {e}")

    # Fallback: script oficial
    log("Instalando via script oficial de Ollama...")
    try:
        result = subprocess.run(
            ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            capture_output=True, text=True, timeout=300
        )
        if find_ollama_executable():
            return True
    except Exception as e:
        log(f"Script oficial falló: {e}")

    return False


def install_ollama_linux() -> bool:
    """Instala Ollama en Linux."""
    log("Instalando Ollama via script oficial...")
    try:
        result = subprocess.run(
            ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
            timeout=300
        )
        if find_ollama_executable():
            log("Ollama instalado correctamente.")
            return True
    except Exception as e:
        log(f"Error: {e}")

    return False


def start_ollama(ollama_cmd: str) -> bool:
    """Arranca ollama serve como proceso de fondo."""
    if is_ollama_running():
        log("Ollama ya está corriendo.")
        return True

    log(f"Iniciando ollama serve ({ollama_cmd})...")
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
        elif IS_MACOS:
            # En macOS, si es Ollama.app, usar open -a
            if "/Ollama.app/" in ollama_cmd:
                subprocess.Popen(
                    ["open", "-a", "Ollama"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                subprocess.Popen(
                    [ollama_cmd, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
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

    # Esperar a que esté listo (máximo 30 segundos)
    for i in range(15):
        time.sleep(2)
        if is_ollama_running():
            log("Ollama está listo y respondiendo.")
            return True
        log(f"Esperando a Ollama... ({i+1}/15)")

    log("AVISO: Ollama no respondió después de 30 segundos.")
    return False


def main():
    log(f"Plataforma: {sys.platform} ({os.name})")

    # 1. Verificar si ya está instalado
    ollama_cmd = find_ollama_executable()

    if ollama_cmd:
        log(f"Ollama encontrado: {ollama_cmd}")
    else:
        # 2. Instalar según plataforma
        log("Ollama no encontrado. Instalando...")
        if IS_WINDOWS:
            success = install_ollama_windows()
        elif IS_MACOS:
            success = install_ollama_macos()
        else:
            success = install_ollama_linux()

        if not success:
            log("ERROR: No se pudo instalar Ollama automáticamente.")
            log("Por favor instala Ollama manualmente desde https://ollama.com/download")
            log("La instalación del plugin continuará, pero necesitarás Ollama para usar la IA.")
            # No hacer sys.exit(1) — el plugin puede funcionar parcialmente sin Ollama
            return

        # Buscar de nuevo después de instalar
        ollama_cmd = find_ollama_executable()
        if not ollama_cmd:
            log("AVISO: Ollama instalado pero no encontrado en PATH.")
            log("Reinicia Pinokio después de la instalación para que detecte Ollama.")
            return

    # 3. Arrancar Ollama
    started = start_ollama(ollama_cmd)
    if started:
        log("Ollama instalado y corriendo correctamente.")
    else:
        log("AVISO: Ollama instalado pero no se pudo iniciar automáticamente.")
        log("Se intentará iniciar al arrancar el plugin.")


if __name__ == "__main__":
    main()
