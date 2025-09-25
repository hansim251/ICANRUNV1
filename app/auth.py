"""Authentication routes for Strava OAuth."""
from __future__ import annotations

import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from . import deps, strava
from .db import get_session
from .models import Credential, Session, User

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _sanitize_next_param(value: Optional[str]) -> str:
    if not value or not value.startswith("/"):
        return "/"
    return value


@router.get("/login")
async def login(
    request: Request,
    next: str = "/",
    db: AsyncSession = Depends(get_session),
) -> RedirectResponse:
    state = secrets.token_urlsafe(32)
    desired_path = _sanitize_next_param(next)

    session_id = deps.read_session_id(request)
    session: Optional[Session] = None
    if session_id:
        result = await db.exec(select(Session).where(Session.id == session_id))
        session = result.one_or_none()
        if session and session.expires_at <= datetime.utcnow():
            await db.delete(session)
            session = None

    if not session:
        session = Session()
        db.add(session)
        await db.commit()
        await db.refresh(session)

    session.state = state
    session.pending_redirect = desired_path
    session.expires_at = datetime.utcnow() + deps.SESSION_TTL
    session.last_seen = datetime.utcnow()
    await db.commit()

    try:
        authorize_url = strava.build_authorize_url(state)
    except strava.StravaConfigError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    redirect = RedirectResponse(authorize_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    deps.set_session_cookie(redirect, session.id)
    return redirect


@router.get("/callback")
async def callback(
    request: Request,
    code: str,
    state: str,
    db: AsyncSession = Depends(get_session),
) -> RedirectResponse:
    session_id = deps.read_session_id(request)
    if not session_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing session")

    session_result = await db.exec(select(Session).where(Session.id == session_id))
    session = session_result.one_or_none()
    if not session or session.state != state:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid OAuth state")

    try:
        token_payload = await strava.exchange_code(code)
    except strava.StravaConfigError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    athlete = token_payload.get("athlete") or {}
    athlete_id = athlete.get("id")
    if not athlete_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Athlete data missing")

    name_parts = [athlete.get("firstname"), athlete.get("lastname")] if athlete else []
    username = " ".join(filter(None, name_parts)) or athlete.get("username") or None

    user_stmt = select(User).where(User.athlete_id == athlete_id)
    user_result = await db.exec(user_stmt)
    user = user_result.one_or_none()
    if user:
        user.username = username or user.username
    else:
        user = User(athlete_id=athlete_id, username=username)
        db.add(user)
        await db.flush()

    credential_stmt = select(Credential).where(Credential.user_id == user.id)
    credential_result = await db.exec(credential_stmt)
    credentials = credential_result.one_or_none()
    expires_at = datetime.utcfromtimestamp(token_payload["expires_at"])

    if credentials:
        credentials.access_token = token_payload["access_token"]
        credentials.refresh_token = token_payload["refresh_token"]
        credentials.expires_at = expires_at
        credentials.scope = token_payload.get("scope")
    else:
        credentials = Credential(
            user_id=user.id,
            access_token=token_payload["access_token"],
            refresh_token=token_payload["refresh_token"],
            expires_at=expires_at,
            scope=token_payload.get("scope"),
        )
        db.add(credentials)

    redirect_target = session.pending_redirect or "/"
    session.user_id = user.id
    session.state = None
    session.pending_redirect = None
    session.expires_at = datetime.utcnow() + deps.SESSION_TTL
    session.last_seen = datetime.utcnow()

    await db.commit()

    redirect = RedirectResponse(redirect_target, status_code=status.HTTP_303_SEE_OTHER)
    deps.set_session_cookie(redirect, session.id)
    return redirect


@router.post("/logout")
async def logout(
    request: Request,
    db: AsyncSession = Depends(get_session),
) -> JSONResponse:
    session_id = deps.read_session_id(request)
    if session_id:
        result = await db.exec(select(Session).where(Session.id == session_id))
        session = result.one_or_none()
        if session:
            await db.delete(session)
            await db.commit()
    payload = JSONResponse({"success": True})
    deps.clear_session_cookie(payload)
    return payload

