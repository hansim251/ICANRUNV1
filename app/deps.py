"""FastAPI dependencies for authentication and session management."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, Request, Response, status
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from .db import get_session
from .models import Credential, Session, User, SESSION_TTL_DAYS

SESSION_COOKIE_NAME = "session_id"
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-insecure-session-secret")
SESSION_TTL = timedelta(days=SESSION_TTL_DAYS)
COOKIE_DOMAIN = os.getenv("COOKIE_DOMAIN") or None
COOKIE_SECURE = os.getenv("COOKIE_SECURE", "1") != "0"

_signer = TimestampSigner(SESSION_SECRET)


def _sign(value: str) -> str:
    return _signer.sign(value.encode()).decode()


def _unsign(value: str) -> Optional[str]:
    try:
        unsigned = _signer.unsign(value.encode(), max_age=int(SESSION_TTL.total_seconds() * 2))
        return unsigned.decode()
    except (BadSignature, SignatureExpired):
        return None


def set_session_cookie(response: Response, session_id: str) -> None:
    """Attach the signed session cookie to the outgoing response."""
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=_sign(session_id),
        max_age=int(SESSION_TTL.total_seconds()),
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="lax",
        domain=COOKIE_DOMAIN,
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        domain=COOKIE_DOMAIN,
    )


def read_session_id(request: Request) -> Optional[str]:
    raw_cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw_cookie:
        return None
    return _unsign(raw_cookie)


async def require_session(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_session),
) -> Session:
    """Resolve and validate the active session from cookies."""
    session_id = read_session_id(request)
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    result = await db.exec(select(Session).where(Session.id == session_id))
    session = result.one_or_none()
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    now = datetime.utcnow()
    if session.expires_at <= now:
        await db.delete(session)
        await db.commit()
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    # Rolling expiry: bump window on each valid request.
    session.last_seen = now
    session.expires_at = now + SESSION_TTL
    await db.commit()
    await db.refresh(session)
    set_session_cookie(response, session.id)
    return session


async def require_user(
    current_session: Session = Depends(require_session),
    db: AsyncSession = Depends(get_session),
) -> tuple[User, Session, Credential]:
    """Return the authenticated user, session, and credentials."""
    if not current_session.user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session not linked")

    user_stmt = select(User).where(User.id == current_session.user_id)
    user_result = await db.exec(user_stmt)
    user = user_result.one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User missing")

    cred_stmt = select(Credential).where(Credential.user_id == user.id)
    cred_result = await db.exec(cred_stmt)
    credentials = cred_result.one_or_none()
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credentials missing")

    return user, current_session, credentials


async def get_current_user(
    deps: tuple[User, Session, Credential] = Depends(require_user),
) -> User:
    user, _, _ = deps
    return user


async def get_user_credentials(
    deps: tuple[User, Session, Credential] = Depends(require_user),
) -> Credential:
    _, _, credentials = deps
    return credentials

