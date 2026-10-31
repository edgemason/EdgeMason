from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

from api.services.artifacts import find_artifact


router = APIRouter()


@router.get("/download/{artifact_id}")
async def download_file(artifact_id: str):
    artifact_path = find_artifact(artifact_id)
    if artifact_path is None:
        raise HTTPException(status_code=404, detail="artifact not found")

    async def content():
        with artifact_path.open("rb") as artifact:
            while chunk := artifact.read(1024 * 1024):
                yield chunk

    async def cleanup() -> None:
        artifact_path.unlink(missing_ok=True)

    return StreamingResponse(
        content(),
        media_type="application/zip",
        headers={
            "Content-Disposition": 'attachment; filename="edgemason-deployment.zip"'
        },
        background=BackgroundTask(cleanup),
    )
