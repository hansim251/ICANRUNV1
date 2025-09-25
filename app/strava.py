"""Helpers for talking to the Strava API."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException, status
from sqlmodel.ext.asyncio.session import AsyncSession

from .models import Credential

STRAVA_CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
STRAVA_CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
SCOPES = os.getenv("SCOPES", "read,activity:read_all")

TOKEN_URL = "https://www.strava.com/oauth/token"
API_BASE = "https://www.strava.com/api/v3"
CLOCK_SKEW = timedelta(seconds=60)


class StravaConfigError(RuntimeError):
    """Raised when required Strava configuration is missing."""


def _require_config() -> None:
    if not STRAVA_CLIENT_ID or not STRAVA_CLIENT_SECRET or not REDIRECT_URI:
        raise StravaConfigError("Strava API credentials are not configured")


def build_authorize_url(state: str) -> str:
    _require_config()
    scope_param = SCOPES.replace(" ", "")
    return (
        "https://www.strava.com/oauth/authorize"
        f"?client_id={STRAVA_CLIENT_ID}"
        f"&response_type=code&redirect_uri={REDIRECT_URI}"
        f"&approval_prompt=auto&scope={scope_param}&state={state}"
    )


def _epoch_to_datetime(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).replace(tzinfo=None)


async def exchange_code(code: str) -> Dict[str, Any]:
    _require_config()
    payload = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(TOKEN_URL, data=payload)
    response.raise_for_status()
    return response.json()


async def refresh_tokens(refresh_token: str) -> Dict[str, Any]:
    _require_config()
    payload = {
        "client_id": STRAVA_CLIENT_ID,
        "client_secret": STRAVA_CLIENT_SECRET,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(TOKEN_URL, data=payload)
    response.raise_for_status()
    return response.json()


async def ensure_valid_credentials(
    db: AsyncSession,
    credentials: Credential,
) -> Credential:
    """Refresh credentials if they are near expiry."""
    now = datetime.utcnow()
    if credentials.expires_at - CLOCK_SKEW > now:
        return credentials

    token_payload = await refresh_tokens(credentials.refresh_token)
    credentials.access_token = token_payload["access_token"]
    credentials.refresh_token = token_payload["refresh_token"]
    credentials.expires_at = _epoch_to_datetime(token_payload["expires_at"])
    credentials.scope = token_payload.get("scope")
    db.add(credentials)
    await db.commit()
    await db.refresh(credentials)
    return credentials


async def _authorized_request(
    db: AsyncSession,
    credentials: Credential,
    method: str,
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> httpx.Response:
    credentials = await ensure_valid_credentials(db, credentials)

    headers = {"Authorization": f"Bearer {credentials.access_token}"}
    async with httpx.AsyncClient(base_url=API_BASE, timeout=20) as client:
        response = await client.request(method, path, params=params, headers=headers)

    if response.status_code == status.HTTP_401_UNAUTHORIZED:
        credentials = await ensure_valid_credentials(db, credentials)
        headers["Authorization"] = f"Bearer {credentials.access_token}"
        async with httpx.AsyncClient(base_url=API_BASE, timeout=20) as client:
            response = await client.request(method, path, params=params, headers=headers)

    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={"error": "strava_api_error", "status": response.status_code},
        )

    return response


async def get_athlete_profile(db: AsyncSession, credentials: Credential) -> Dict[str, Any]:
    response = await _authorized_request(db, credentials, "GET", "/athlete")
    return response.json()


async def get_activities(
    db: AsyncSession,
    credentials: Credential,
    *,
    after: Optional[datetime] = None,
    before: Optional[datetime] = None,
    per_page: int = 30,
    page: int = 1,
) -> list[Dict[str, Any]]:
    params: Dict[str, Any] = {"per_page": per_page, "page": page}
    if after:
        params["after"] = int(after.replace(tzinfo=timezone.utc).timestamp())
    if before:
        params["before"] = int(before.replace(tzinfo=timezone.utc).timestamp())

    response = await _authorized_request(db, credentials, "GET", "/athlete/activities", params=params)
    return response.json()


