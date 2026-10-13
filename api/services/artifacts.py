import time
from pathlib import Path
from uuid import UUID, uuid4

from api.config import ARTIFACT_DIR, ARTIFACT_TTL_SECONDS


def new_artifact_path() -> tuple[str, Path]:
    artifact_id = uuid4().hex
    return artifact_id, ARTIFACT_DIR / f"{artifact_id}.zip"


def find_artifact(artifact_id: str) -> Path | None:
    try:
        parsed_id = UUID(hex=artifact_id).hex
    except ValueError:
        return None
    path = ARTIFACT_DIR / f"{parsed_id}.zip"
    return path if path.is_file() else None


def cleanup_stale_artifacts() -> None:
    cutoff = time.time() - ARTIFACT_TTL_SECONDS
    for path in ARTIFACT_DIR.glob("*.zip"):
        if path.is_file() and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
