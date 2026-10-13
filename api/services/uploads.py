from pathlib import Path

from fastapi import HTTPException, UploadFile


def require_suffix(upload: UploadFile, suffixes: set[str], detail: str) -> str:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in suffixes:
        raise HTTPException(status_code=422, detail=detail)
    return suffix


def save_upload(upload: UploadFile, path: Path, limit: int) -> None:
    size = 0
    try:
        with path.open("wb") as destination:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(status_code=413, detail="uploaded file is too large")
                destination.write(chunk)
    finally:
        upload.file.close()
