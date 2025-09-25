"""Routes for authenticated user profile data."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_current_user
from ..models import User

router = APIRouter(prefix="/api", tags=["me"])


@router.get("/me")
async def read_me(user: User = Depends(get_current_user)) -> dict[str, int | str | None]:
    """Return the currently authenticated athlete."""
    return {
        "athlete_id": user.athlete_id,
        "username": user.username,
    }
