"""
pull_models.py — Descarga los modelos de IA necesarios via Ollama API.
Ejecutado por install.json. Funciona en Windows, macOS y Linux.

Usa la API HTTP de Ollama (no el CLI) para evitar problemas de PATH en Windows.
Los modelos se descargan uno por uno con progreso visible.
"""

import json
import sys
import time
import urllib.request
import urllib.error

OLLAMA_URL = "http://127.0.0.1:11434"

MODELS = [
    {
        "name": "qwen3.5:0.8b",
        "description": "Agente de lenguaje para extraccion y clasificacion (~500MB)",
    },
    {
        "name": "moondream",
        "description": "Agente visual para leer documentos escaneados (~1.8GB)",
    },
]


def log(msg: str):
    print(f"[pull_models] {msg}", flush=True)


def is_ollama_running() -> bool:
    """Verifica si Ollama está respondiendo."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_model_available(model_name: str) -> bool:
    """Verifica si un modelo ya está descargado."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in data.get("models", [])]
            base_name = model_name.split(":")[0]
            return model_name in models or any(m.startswith(base_name) for m in models)
    except Exception:
        return False


def pull_model(model_name: str) -> bool:
    """Descarga un modelo via la API HTTP de Ollama (streaming)."""
    log(f"Descargando {model_name}...")

    try:
        payload = json.dumps({"name": model_name, "stream": True}).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        last_progress = -1
        with urllib.request.urlopen(req, timeout=3600) as resp:
            buffer = b""
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                buffer += chunk
                # Procesar líneas completas
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                        status = data.get("status", "")
                        completed = data.get("completed", 0)
                        total = data.get("total", 0)

                        if total > 0:
                            pct = int(completed / total * 100)
                            if pct != last_progress and pct % 10 == 0:
                                last_progress = pct
                                size_mb = total / (1024 * 1024)
                                log(f"  {model_name}: {pct}% ({size_mb:.0f}MB)")
                        elif status:
                            log(f"  {status}")
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass

        log(f"Modelo {model_name} descargado correctamente.")
        return True

    except urllib.error.HTTPError as e:
        log(f"ERROR HTTP {e.code} descargando {model_name}: {e.reason}")
        return False
    except Exception as e:
        log(f"ERROR descargando {model_name}: {e}")
        return False


def main():
    if not is_ollama_running():
        log("AVISO: Ollama no está corriendo en " + OLLAMA_URL)
        log("Los modelos se descargarán automáticamente cuando inicies el plugin.")
        log("Si quieres descargarlos ahora, asegúrate de que Ollama esté corriendo.")
        sys.exit(0)  # No fallar — el plugin puede funcionar sin modelos pre-descargados

    all_ok = True
    for model_info in MODELS:
        name = model_info["name"]
        desc = model_info["description"]

        if is_model_available(name):
            log(f"{name} ya está disponible.")
            continue

        log(f"Descargando: {desc}")
        ok = pull_model(name)
        if not ok:
            log(f"AVISO: No se pudo descargar {name}. Se intentará al usar el plugin.")
            all_ok = False

    if all_ok:
        log("Todos los modelos están listos.")
    else:
        log("Algunos modelos no se pudieron descargar. Se intentará al iniciar el plugin.")


if __name__ == "__main__":
    main()
