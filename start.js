module.exports = {
  daemon: true,
  run: [
    // Verificar que Ollama esté corriendo
    {
      method: "shell.run",
      params: {
        message: "curl -s http://localhost:11434/api/tags > /dev/null || ollama serve &",
        background: true
      }
    },
    {
      method: "shell.run",
      params: {
        message: "sleep 2"
      }
    },
    
    // Iniciar servidor backend
    {
      method: "shell.run",
      params: {
        message: "python server/app.py",
        path: "{{cwd}}",
        venv: "venv",
        env: {
          PORT: "{{port}}",
          DATA_DIR: "{{cwd}}/data",
          PLUGIN_DIR: "{{cwd}}"
        }
      }
    }
  ]
}
