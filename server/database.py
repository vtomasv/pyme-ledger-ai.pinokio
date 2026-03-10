"""
Gestión de base de datos SQLite para pyme-ledger-ai.
Inicialización, sesiones y utilidades de persistencia.
"""
import os
import json
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from models import Base

# Configuración — ruta absoluta desde __file__ como fallback
# database.py está en server/, el plugin está en server/../
_BASE_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = Path(os.environ.get("DATA_DIR", str(_BASE_DIR / "data")))
DB_PATH = DATA_DIR / "pyme_ledger.db"

# Crear directorio si no existe
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Crear engine SQLite
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False
)

# Crear session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Inicializa la base de datos creando todas las tablas."""
    Base.metadata.create_all(bind=engine)
    print(f"✅ Base de datos inicializada en: {DB_PATH}")


def get_db() -> Session:
    """Obtiene una sesión de base de datos."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_json(path: Path, data: dict):
    """Guarda un diccionario como JSON con UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path, default=None) -> dict:
    """Carga un JSON desde archivo."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error cargando JSON {path}: {e}")
            return default or {}
    return default or {}


def save_list_json(path: Path, data: list):
    """Guarda una lista como JSON con UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_list_json(path: Path, default=None) -> list:
    """Carga una lista JSON desde archivo."""
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Error cargando JSON {path}: {e}")
            return default or []
    return default or []
