"""
launcher.py — Lanzador cross-platform para pyme-ledger-ai.
Ejecutado por start.json. Detecta la plataforma y usa la ruta
correcta del Python del venv para iniciar server/app.py.

Funciona en: macOS, Windows, Linux.
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
IS_WINDOWS = sys.platform == "win32"

# Ruta al Python del venv según plataforma
VENV_PYTHON = BASE_DIR / "venv" / (
    "Scripts/python.exe" if IS_WINDOWS else "bin/python"
)

SERVER_SCRIPT = BASE_DIR / "server" / "app.py"

def main():
    if not VENV_PYTHON.exists():
        print(f"ERROR: Venv no encontrado en {VENV_PYTHON}", flush=True)
        print("Por favor reinstala el plugin desde Pinokio.", flush=True)
        sys.exit(1)

    if not SERVER_SCRIPT.exists():
        print(f"ERROR: server/app.py no encontrado en {SERVER_SCRIPT}", flush=True)
        sys.exit(1)

    # Heredar todas las variables de entorno (PORT, DATA_DIR, etc. inyectadas por Pinokio)
    env = os.environ.copy()

    # Garantizar que DATA_DIR y PLUGIN_DIR usen rutas absolutas
    if "DATA_DIR" not in env or not env["DATA_DIR"]:
        env["DATA_DIR"] = str(BASE_DIR / "data")
    if "PLUGIN_DIR" not in env or not env["PLUGIN_DIR"]:
        env["PLUGIN_DIR"] = str(BASE_DIR)
    if "OLLAMA_URL" not in env:
        env["OLLAMA_URL"] = "http://localhost:11434"

    print(f"[launcher] Plataforma: {sys.platform}", flush=True)
    print(f"[launcher] Python venv: {VENV_PYTHON}", flush=True)
    print(f"[launcher] Servidor: {SERVER_SCRIPT}", flush=True)
    print(f"[launcher] PORT: {env.get('PORT', '8000')}", flush=True)
    print(f"[launcher] DATA_DIR: {env['DATA_DIR']}", flush=True)

    # Ejecutar server/app.py con el Python del venv
    # subprocess.run reemplaza el proceso actual (exec-style)
    result = subprocess.run(
        [str(VENV_PYTHON), str(SERVER_SCRIPT)],
        env=env,
        cwd=str(BASE_DIR)
    )
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
