"""FastAPI application entry point for the Strava dashboard."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from . import auth, deps
from .db import get_session, init_db
from .models import Session
from .routers import activities, me, stats

load_dotenv()

TITLE = "ICANRUN Strava Dashboard"
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
LANDING_FILE = STATIC_DIR / "landing.html"
DASHBOARD_FILE = STATIC_DIR / "dashboard.html"

app = FastAPI(title=TITLE, version="2.0.0")

app.include_router(auth.router)
app.include_router(me.router)
app.include_router(activities.router)
app.include_router(stats.router)

FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")
if FRONTEND_ORIGIN:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_ORIGIN],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
async def _startup() -> None:
    await init_db()


@app.get("/")
async def index(request: Request, db: AsyncSession = Depends(get_session)) -> FileResponse:
    session_id = deps.read_session_id(request)
    if not session_id:
        landing = FileResponse(LANDING_FILE)
        deps.clear_session_cookie(landing)
        return landing

    result = await db.exec(select(Session).where(Session.id == session_id))
    session = result.one_or_none()
    if not session or session.expires_at <= datetime.utcnow() or not session.user_id:
        if session:
            await db.delete(session)
            await db.commit()
        landing = FileResponse(LANDING_FILE)
        deps.clear_session_cookie(landing)
        return landing

    session.last_seen = datetime.utcnow()
    session.expires_at = session.last_seen + deps.SESSION_TTL
    await db.commit()

    dashboard = FileResponse(DASHBOARD_FILE)
    deps.set_session_cookie(dashboard, session.id)
    return dashboard


@app.get("/healthz")
def healthcheck() -> JSONResponse:
    return JSONResponse({"status": "ok"})

