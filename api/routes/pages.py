from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from api.config import STATIC_INDEX


router = APIRouter()


def _index(fallback: str) -> HTMLResponse:
    if STATIC_INDEX.exists():
        return HTMLResponse(content=STATIC_INDEX.read_text(encoding="utf-8"))
    return HTMLResponse(fallback)


@router.get("/", response_class=HTMLResponse)
async def root():
    return _index("<h1>EdgeForge API</h1>")


@router.get("/ui", response_class=HTMLResponse)
async def ui():
    return _index("<h1>EdgeForge UI</h1>")


@router.get("/api")
async def api_info():
    return {"message": "EdgeMason API"}


@router.get("/health")
async def health():
    return {"status": "ok"}
