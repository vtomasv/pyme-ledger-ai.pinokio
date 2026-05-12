const path = require("path")
const fs = require("fs")

module.exports = {
  title: "CCS — Pyme Ledger AI",
  description: "Clasificación inteligente de gastos con IA local para PyMEs — 100% offline | CCS",
  icon: "icon.png",
  version: "1.4.0",

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

    // Verificar si está corriendo usando la info que Pinokio provee
    const running = info && info.running

    // ---- Estado: No instalado ----
    if (!installed) {
      return [{
        default: true,
        icon: "fa-solid fa-download",
        text: "Instalar",
        href: "install.json",
        description: "Instalación automática con 1 click (5-15 min)"
      }]
    }

    // ---- Estado: Corriendo ----
    if (running) {
      return [
        {
          icon: "fa-solid fa-circle",
          text: "En ejecución",
          href: "start.json",
          style: "color: #3DAE2B"
        },
        {
          icon: "fa-solid fa-stop",
          text: "Detener",
          href: "stop.json"
        }
      ]
    }

    // ---- Estado: Instalado pero no corriendo ----
    return [
      {
        default: true,
        icon: "fa-solid fa-play",
        text: "Iniciar",
        href: "start.json",
        description: "Iniciar el plugin"
      },
      {
        icon: "fa-solid fa-rotate",
        text: "Reinstalar",
        href: "install.json",
        description: "Reinstalar si hay problemas"
      }
    ]
  }
}
