from fastapi import FastAPI

from api.routes import analyze, artifacts, deploy, pages


app = FastAPI(title="EdgeMason", version="0.1.0")
app.include_router(pages.router)
app.include_router(analyze.router)
app.include_router(deploy.router)
app.include_router(artifacts.router)
