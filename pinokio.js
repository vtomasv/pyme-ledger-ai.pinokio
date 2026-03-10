module.exports = {
  title: "Pyme Ledger AI",
  description: "Gestión de documentos contables con IA local - Extracción, clasificación y análisis 100% offline",
  icon: "icon.png",
  version: "1.0.0",
  
  menu: async (kernel, info) => {
    // Verificar si está instalado
    const installed = await kernel.exists(__dirname, "venv")
    const running = await kernel.script.running(__dirname, "start.json")
    
    if (!installed) {
      return [{
        default: true,
        icon: "fa-solid fa-download",
        text: "Instalar",
        href: "install.json",
        description: "Instalación automática con 1 click"
      }]
    }
    
    if (running) {
      return [
        {
          icon: "fa-solid fa-circle",
          text: "En ejecución",
          href: "start.json",
          style: "color: #22c55e"
        },
        {
          icon: "fa-solid fa-stop",
          text: "Detener",
          href: "stop.json"
        }
      ]
    }
    
    return [
      {
        default: true,
        icon: "fa-solid fa-play",
        text: "Iniciar",
        href: "start.json",
        description: "Iniciar el plugin"
      }
    ]
  }
}
