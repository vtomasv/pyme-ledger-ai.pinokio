module.exports = {
  title: "Instalando Plugin",
  description: "Instalación automática de dependencias",
  run: [
    // Fase 1: Verificación del sistema
    {
      method: "log",
      params: {
        html: "<div style='padding:16px'><h3>🔍 Verificando sistema...</h3></div>"
      }
    },
    
    // Fase 2: Instalar/verificar Ollama
    {
      method: "log",
      params: {
        html: "<div style='padding:16px'><h3>🤖 Configurando Ollama...</h3></div>"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "ollama --version || (curl -fsSL https://ollama.com/install.sh | sh)",
        on: [{
          event: "error",
          done: false
        }]
      }
    },
    
    // Iniciar Ollama en background
    {
      method: "shell.run",
      params: {
        message: "ollama serve",
        background: true
      }
    },
    {
      method: "shell.run",
      params: {
        message: "sleep 3"
      }
    },
    
    // Fase 3: Descargar modelo según RAM disponible
    {
      method: "log",
      params: {
        html: "<div style='padding:16px'><h3>⬇️ Descargando modelo de IA...</h3><p style='color:#94a3b8'>Esto puede tomar varios minutos la primera vez.</p></div>"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "{{ram < 8 ? 'ollama pull llama3.2:1b' : ram < 16 ? 'ollama pull llama3.2:3b' : 'ollama pull llama3.1:8b'}}"
      }
    },
    
    // Fase 4: Entorno Python
    {
      method: "log",
      params: {
        html: "<div style='padding:16px'><h3>🐍 Configurando entorno Python...</h3></div>"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "python -m venv venv",
        path: "{{cwd}}"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "pip install --upgrade pip && pip install -r requirements.txt",
        path: "{{cwd}}",
        venv: "venv"
      }
    },
    
    // Fase 5: Inicializar datos
    {
      method: "log",
      params: {
        html: "<div style='padding:16px'><h3>💾 Inicializando datos...</h3></div>"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "mkdir -p data/agents data/prompts/system data/prompts/templates data/sessions data/exports",
        path: "{{cwd}}"
      }
    },
    {
      method: "fs.write",
      params: {
        path: "data/config.json",
        text: "{{JSON.stringify({version: '1.0.0', installedAt: new Date().toISOString(), defaultModel: ram < 8 ? 'llama3.2:1b' : ram < 16 ? 'llama3.2:3b' : 'llama3.1:8b'}, null, 2)}}"
      }
    },
    {
      method: "shell.run",
      params: {
        message: "cp defaults/agents.json data/agents/agents.json 2>/dev/null || true && cp -r defaults/prompts/* data/prompts/ 2>/dev/null || true",
        path: "{{cwd}}"
      }
    },
    
    // Completado
    {
      method: "log",
      params: {
        html: "<div style='padding:16px;background:#0f2d1a;border-radius:8px;margin:16px'><h3 style='color:#22c55e'>✅ Instalación completada</h3><p>Haz click en 'Iniciar' para comenzar a usar el plugin.</p></div>"
      }
    },
    {
      method: "notify",
      params: {
        html: "Plugin instalado correctamente. Haz click en 'Iniciar' para comenzar."
      }
    }
  ]
}
