from pathlib import Path

#: Ruta absoluta a la raíz del proyecto (independiente del directorio actual).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

__all__ = ["PROJECT_ROOT"]
