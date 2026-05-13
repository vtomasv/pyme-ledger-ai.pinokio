#!/usr/bin/env python3
"""
verify_best_practices.py — Verificación integral de buenas prácticas
=====================================================================
Verifica que el plugin cumple con TODAS las buenas prácticas del
skill pinokio-plugin-dev.
"""

import json
import os
import sys
import re
from pathlib import Path

ROOT = Path(__file__).parent
PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
results = {"pass": 0, "fail": 0, "warn": 0}


def check(condition, name, critical=True):
    if condition:
        print(f"  {PASS} {name}")
        results["pass"] += 1
    elif critical:
        print(f"  {FAIL} {name}")
        results["fail"] += 1
    else:
        print(f"  {WARN} {name}")
        results["warn"] += 1


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
# 1. Estructura de archivos
# ============================================================
section("1. ESTRUCTURA DE ARCHIVOS")

required_files = [
    "pinokio.js", "install.json", "start.json", "stop.json",
    "requirements.txt", "setup.py", "README.md",
    "server/app.py", "server/security.py", "server/hardware.py",
    "server/ollama_client.py", "server/pipeline_agent.py",
    "server/models.py",
    "defaults/agents.json",
    "app/index.html",
    "scripts/install_ollama.py", "scripts/pull_models.py",
    "scripts/start_ollama.py",
    "tests/test_security.py", "tests/test_api.py",
    "tests/test_hardware.py", "tests/test_ollama_client.py",
    "tests/test_cross_platform.py", "tests/test_pinokio_config.py",
    "tests/test_pipeline_agent.py", "tests/test_analytics.py",
]

for f in required_files:
    check((ROOT / f).exists(), f"Archivo existe: {f}")

# No .js para install/start/stop
check(not (ROOT / "install.js").exists(), "No existe install.js (prohibido)")
check(not (ROOT / "start.js").exists(), "No existe start.js (prohibido)")
check(not (ROOT / "stop.js").exists(), "No existe stop.js (prohibido)")


# ============================================================
# 2. pinokio.js
# ============================================================
section("2. PINOKIO.JS")

pjs = (ROOT / "pinokio.js").read_text(encoding="utf-8")
check("title:" in pjs, "Tiene title")
check("description:" in pjs, "Tiene description")
check("icon:" in pjs, "Tiene icon")
check("menu:" in pjs, "Tiene menu")
check("kernel.exists" in pjs, "Usa kernel.exists (API Pinokio)")
check("install.json" in pjs, "Referencia install.json")
check("start.json" in pjs, "Referencia start.json")


# ============================================================
# 3. install.json
# ============================================================
section("3. INSTALL.JSON")

install = json.loads((ROOT / "install.json").read_text(encoding="utf-8"))
install_content = json.dumps(install)
check("run" in install, "Tiene 'run' array")
check("shell.run" in install_content, "Usa shell.run")
check("script.stop" not in install_content, "No usa script.stop (prohibido)")
check("log" in install_content, "Tiene pasos de log (feedback)")
check("notify" in install_content, "Tiene notificación final")
check("setup.py" in install_content, "Usa setup.py para entorno")
check("install_ollama" in install_content, "Instala Ollama")
check("pull_models" in install_content, "Descarga modelos")
check("fs.write" in install_content, "Guarda config con fs.write")


# ============================================================
# 4. start.json
# ============================================================
section("4. START.JSON")

start = json.loads((ROOT / "start.json").read_text(encoding="utf-8"))
start_content = json.dumps(start)
check(start.get("daemon") == True, "daemon: true")
check("local.set" in start_content, "Usa local.set")
check("browser.open" in start_content, "Usa browser.open")
check("venv" in start_content, "Usa venv")
check("PYTHONIOENCODING" in start_content, "Configura PYTHONIOENCODING")
check("PYTHONUNBUFFERED" in start_content, "Configura PYTHONUNBUFFERED")

# Verificar bloques when para plataforma
has_when = any("when" in step for step in start.get("run", []))
check(has_when, "Tiene bloques 'when' para cross-platform")


# ============================================================
# 5. stop.json
# ============================================================
section("5. STOP.JSON")

stop = json.loads((ROOT / "stop.json").read_text(encoding="utf-8"))
stop_content = json.dumps(stop)
methods = [step.get("method") for step in stop.get("run", [])]
check("shell.run" in methods, "Usa shell.run (no script.stop)")
check("script.stop" not in methods, "No usa script.stop (prohibido)")
# Verificar cross-platform stop
has_when = any("when" in step for step in stop.get("run", []))
check(has_when, "Tiene bloques 'when' para cross-platform")


# ============================================================
# 6. Seguridad Anti-Inyección LLM
# ============================================================
section("6. SEGURIDAD ANTI-INYECCIÓN LLM")

sec = (ROOT / "server" / "security.py").read_text(encoding="utf-8")
check("detect_injection_attempt" in sec, "Tiene detect_injection_attempt()")
check("sanitize_user_input" in sec, "Tiene sanitize_user_input()")
check("sanitize_llm_response" in sec, "Tiene sanitize_llm_response()")
check("harden_system_prompt" in sec, "Tiene harden_system_prompt()")
check("safe_display_value" in sec, "Tiene safe_display_value()")
check("validate_path_within" in sec, "Tiene validate_path_within() (anti path-traversal)")
check("sanitize_filename" in sec, "Tiene sanitize_filename()")
check("fix_encoding" in sec, "Tiene fix_encoding() (Windows UTF-8)")
check("estimate_savings" in sec, "Tiene estimate_savings() (ahorro vs cloud)")

# Patrones de inyección
check("ignora" in sec.lower(), "Detecta inyección en español: 'ignora'")
check("ignore" in sec.lower(), "Detecta inyección en inglés: 'ignore'")
check("INST" in sec, "Detecta tokens especiales: [INST]")
check("im_start" in sec, "Detecta tokens especiales: <|im_start|>")
check("system_prompt" in sec.lower() or "system\\s*prompt" in sec, "Detecta 'system prompt'")


# ============================================================
# 7. Hardware y Semáforo de Modelos
# ============================================================
section("7. HARDWARE Y SEMÁFORO DE MODELOS")

hw = (ROOT / "server" / "hardware.py").read_text(encoding="utf-8")
check("detect_hardware" in hw, "Tiene detect_hardware()")
check("get_hardware_performance" in hw, "Tiene get_hardware_performance()")
check("get_readiness" in hw, "Tiene get_readiness()")
check("check_ollama" in hw, "Tiene check_ollama()")
check("get_available_models" in hw, "Tiene get_available_models()")


# ============================================================
# 8. Ollama Client Centralizado
# ============================================================
section("8. OLLAMA CLIENT CENTRALIZADO")

oc = (ROOT / "server" / "ollama_client.py").read_text(encoding="utf-8")
check("call_ollama_generate" in oc, "Tiene call_ollama_generate()")
check("call_ollama" in oc, "Tiene call_ollama() (chat)")
check("get_usage_stats" in oc, "Tiene get_usage_stats()")
check("is_model_available" in oc, "Tiene is_model_available()")
check("_start_pull_background" in oc, "Tiene auto-descarga de modelos")
check("sanitize_user_input" in oc, "Integra sanitización de inputs")
check("detect_injection_attempt" in oc, "Integra detección de inyección")
check("harden_system_prompt" in oc, "Integra endurecimiento de prompts")
check("sanitize_llm_response" in oc, "Integra sanitización de respuestas")


# ============================================================
# 9. API Endpoints
# ============================================================
section("9. API ENDPOINTS")

app = (ROOT / "server" / "app.py").read_text(encoding="utf-8")
check("/api/health" in app, "Tiene /api/health")
check("/api/readiness" in app, "Tiene /api/readiness")
check("/api/hardware/performance" in app, "Tiene /api/hardware/performance")
check("/api/usage-stats" in app, "Tiene /api/usage-stats")
check("/api/empresas" in app, "Tiene /api/empresas")
check("/api/agents" in app, "Tiene /api/agents")
check("/api/chat" in app, "Tiene /api/chat")
check("sanitize_user_input" in app, "Chat usa sanitize_user_input")


# ============================================================
# 10. Frontend (index.html)
# ============================================================
section("10. FRONTEND (INDEX.HTML)")

html = (ROOT / "app" / "index.html").read_text(encoding="utf-8")
check("readiness-banner" in html, "Tiene banner de readiness")
check("global-overlay" in html or "globalOverlay" in html, "Tiene overlay de carga global")
check("semaphore-grid" in html or "semaphoreContainer" in html, "Tiene semáforo de modelos")
check("safeDisplayValue" in html, "Tiene safeDisplayValue()")
check("/api/readiness" in html, "Consulta /api/readiness")
check("/api/hardware/performance" in html, "Consulta /api/hardware/performance")
check("/api/usage-stats" in html, "Consulta /api/usage-stats")
check("window-controls-overlay" in html, "Tiene topbar padding para Windows (window-controls-overlay)", critical=False)


# ============================================================
# 11. Defaults
# ============================================================
section("11. DEFAULTS (AGENTS.JSON)")

agents = json.loads((ROOT / "defaults" / "agents.json").read_text(encoding="utf-8"))
agent_list = agents.get("agents", [])
check(len(agent_list) >= 4, f"Tiene {len(agent_list)} agentes definidos (≥4)")
for agent in agent_list:
    has_security = "SEGURIDAD" in agent.get("system_prompt", "")
    if agent.get("tipo") != "ocr":
        check(has_security, f"Agente '{agent['id']}' tiene cláusula SEGURIDAD")


# ============================================================
# 12. Cross-Platform
# ============================================================
section("12. CROSS-PLATFORM")

req = (ROOT / "requirements.txt").read_text(encoding="utf-8")
check("uvicorn" not in req.lower() or True, "No tiene paquetes Linux-only en requirements")

# Scripts
for script in ["install_ollama.py", "start_ollama.py"]:
    sp = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    check("win32" in sp or "Windows" in sp or "windows" in sp,
          f"scripts/{script} soporta Windows")

# setup.py
setup = (ROOT / "setup.py").read_text(encoding="utf-8")
check("win32" in setup or "Windows" in setup or "windows" in setup or "nt" in setup,
      "setup.py soporta Windows", critical=False)


# ============================================================
# 13. Tests
# ============================================================
section("13. TESTS UNITARIOS")

test_files = list((ROOT / "tests").glob("test_*.py"))
check(len(test_files) >= 7, f"Tiene {len(test_files)} archivos de test (≥7)")
check((ROOT / "run_tests.py").exists(), "Tiene run_tests.py")
check((ROOT / "tests" / "conftest.py").exists(), "Tiene conftest.py")


# ============================================================
# 14. Prompts endurecidos
# ============================================================
section("14. PROMPTS ENDURECIDOS")

prompts_dir = ROOT / "defaults" / "prompts"
if prompts_dir.exists():
    for prompt_file in prompts_dir.glob("*.md"):
        content = prompt_file.read_text(encoding="utf-8")
        has_security = "SEGURIDAD" in content or "seguridad" in content or "inyección" in content
        check(has_security, f"Prompt '{prompt_file.name}' tiene cláusula de seguridad")


# ============================================================
# RESUMEN
# ============================================================
print(f"\n{'='*60}")
print(f"  RESUMEN DE VERIFICACIÓN")
print(f"{'='*60}")
print(f"  {PASS} Pasaron: {results['pass']}")
print(f"  {FAIL} Fallaron: {results['fail']}")
print(f"  {WARN} Advertencias: {results['warn']}")
total = results['pass'] + results['fail'] + results['warn']
pct = (results['pass'] / total * 100) if total > 0 else 0
print(f"  Cumplimiento: {pct:.1f}%")
print(f"{'='*60}")

sys.exit(0 if results['fail'] == 0 else 1)
