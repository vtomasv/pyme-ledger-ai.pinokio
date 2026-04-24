"""
start_ollama.py — Arranca Ollama serve como proceso de fondo.
Ejecutado por start.json antes de iniciar el servidor.
Funciona en Windows, macOS y Linux.

Usa subprocess.Popen para no bloquear el proceso principal.
En Windows usa CREATE_NO_WINDOW para evitar ventanas extra.
"""

import os
import sys
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
OLLAMA_URL = "http://127.0.0.1:11434"


def log(msg: str):
    print(f"[start_ollama] {msg}", flush=True)


def is_ollama_running() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_ollama() -> str:
    """Encuentra el ejecutable de ollama."""
    # Primero buscar en PATH
    ollama = shutil.which("ollama")
    if ollama:
        return ollama

    # En Windows, buscar en rutas comunes
    if IS_WINDOWS:
        common_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path(os.environ.get("PROGRAMFILES", "")) / "Ollama" / "ollama.exe",
            Path(os.environ.get("USERPROFILE", "")) / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        ]
        for p in common_paths:
            if p.exists():
                # Agregar al PATH de la sesión
                ollama_dir = str(p.parent)
                os.environ["PATH"] = ollama_dir + os.pathsep + os.environ.get("PATH", "")
                return str(p)

    return ""


def main():
    if is_ollama_running():
        log("Ollama ya está corriendo.")
        return

    ollama_cmd = find_ollama()
    if not ollama_cmd:
        log("AVISO: Ollama no encontrado. Algunas funciones de IA no estarán disponibles.")
        log("Instala Ollama desde https://ollama.com/download")
        return

    log(f"Iniciando ollama serve ({ollama_cmd})...")
    try:
        if IS_WINDOWS:
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
        log(f"Error iniciando ollama: {e}")
        return

    # Esperar hasta 20 segundos
    for i in range(10):
        time.sleep(2)
        if is_ollama_running():
            log("Ollama listo.")
            return
        log(f"Esperando... ({i+1}/10)")

    log("AVISO: Ollama no respondió. Continuando de todas formas.")


if __name__ == "__main__":
    main()
