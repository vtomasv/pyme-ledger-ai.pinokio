const path = require("path")
const fs = require("fs")

module.exports = {
  title: "CCS — Pyme Ledger AI",
  description: "Clasificación inteligente de gastos con IA local para PyMEs — 100% offline | CCS",
  icon: "icon.png",
  version: "1.3.0",

  menu: async (kernel, info) => {
    // Verificar instalación: comprobar que el venv existe (cross-platform)
    const isWin = process.platform === "win32"
    const venvMarker = isWin
      ? path.resolve(__dirname, "venv", "Scripts", "python.exe")
      : path.resolve(__dirname, "venv", "bin", "python")

    let installed = false
    try {
      installed = fs.existsSync(venvMarker)
    } catch (e) {
      installed = false
    }

    const running = info && info.running

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
