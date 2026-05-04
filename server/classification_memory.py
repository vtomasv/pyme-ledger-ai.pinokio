"""
Módulo de memoria de clasificaciones humanas.
Persiste las correcciones de clasificación del usuario en disco
y las inyecta como contexto en el prompt del clasificador.
"""
import json
from pathlib import Path
from datetime import datetime

# Directorio de datos (se configura al importar)
_DATA_DIR = Path(__file__).parent.parent / "data"


def set_data_dir(data_dir: Path):
    """Configura el directorio de datos."""
    global _DATA_DIR
    _DATA_DIR = data_dir


def save_classification_learning(
    empresa_id: str, proveedor: str, descripcion: str,
    tipo_documento: str, new_cat_name: str, new_categoria_id: str
):
    """
    Guarda la corrección de clasificación del usuario como aprendizaje.
    Se persiste en un archivo JSON que el clasificador usa como contexto
    para futuras clasificaciones similares.
    """
    # Archivo de aprendizaje por empresa
    learn_dir = _DATA_DIR / "learning"
    learn_dir.mkdir(parents=True, exist_ok=True)
    learn_file = learn_dir / f"classifications_{empresa_id}.json"

    # Cargar aprendizajes existentes
    learnings = []
    if learn_file.exists():
        try:
            learnings = json.loads(learn_file.read_text(encoding="utf-8"))
        except Exception:
            learnings = []

    # Normalizar proveedor para búsqueda
    proveedor_norm = proveedor.strip().lower()[:100] if proveedor else ""
    desc_norm = descripcion.strip().lower()[:200] if descripcion else ""

    # Buscar si ya existe un aprendizaje para este proveedor+tipo
    found = False
    for entry in learnings:
        if (entry.get("proveedor_norm") == proveedor_norm and
            entry.get("tipo_documento") == tipo_documento):
            # Actualizar la categoría
            entry["categoria"] = new_cat_name
            entry["categoria_id"] = new_categoria_id
            entry["veces_corregido"] = entry.get("veces_corregido", 0) + 1
            entry["ultima_correccion"] = datetime.utcnow().isoformat()
            found = True
            break

    if not found:
        learnings.append({
            "proveedor": proveedor[:100] if proveedor else "",
            "proveedor_norm": proveedor_norm,
            "descripcion_ejemplo": desc_norm[:200],
            "tipo_documento": tipo_documento,
            "categoria": new_cat_name,
            "categoria_id": new_categoria_id,
            "veces_corregido": 1,
            "fecha_primera": datetime.utcnow().isoformat(),
            "ultima_correccion": datetime.utcnow().isoformat()
        })

    # Guardar (máximo 500 entradas, las más recientes primero)
    learnings.sort(key=lambda x: x.get("ultima_correccion", ""), reverse=True)
    learnings = learnings[:500]
    learn_file.write_text(
        json.dumps(learnings, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"INFO: Aprendizaje guardado — '{proveedor}' → '{new_cat_name}' (total: {len(learnings)})")


def get_classification_learnings(empresa_id: str) -> str:
    """
    Retorna las clasificaciones aprendidas como texto para inyectar en el prompt
    del clasificador. Solo retorna las más relevantes (top 30).
    """
    learn_file = _DATA_DIR / "learning" / f"classifications_{empresa_id}.json"
    if not learn_file.exists():
        return ""

    try:
        learnings = json.loads(learn_file.read_text(encoding="utf-8"))
    except Exception:
        return ""

    if not learnings:
        return ""

    # Ordenar por veces_corregido (más frecuentes primero) y tomar top 30
    learnings.sort(key=lambda x: x.get("veces_corregido", 0), reverse=True)
    top = learnings[:30]

    lines = ["=== CLASIFICACIONES APRENDIDAS DEL USUARIO ==="]
    lines.append("El usuario ha corregido estas clasificaciones previamente. RESPETA estas decisiones:")
    for entry in top:
        prov = entry.get("proveedor", "")
        cat = entry.get("categoria", "")
        tipo = entry.get("tipo_documento", "")
        veces = entry.get("veces_corregido", 1)
        desc = entry.get("descripcion_ejemplo", "")
        line = f"- Proveedor '{prov}'"
        if tipo:
            line += f" (tipo: {tipo})"
        if desc:
            line += f" [ej: {desc[:60]}]"
        line += f" → Categoría: '{cat}' (corregido {veces}x)"
        lines.append(line)
    lines.append("=== FIN CLASIFICACIONES APRENDIDAS ===")
    return "\n".join(lines)
