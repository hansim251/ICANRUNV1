"""Routes that proxy Strava activities for the logged-in user."""
from __future__ import annotations

import json
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from .. import deps, strava
from ..db import get_session
from ..models import ActivityCache, Credential, Session, User

router = APIRouter(prefix="/api", tags=["activities"])


@router.get("/activities")
async def list_activities(
    after: datetime | None = Query(None, description="Filter activities after this timestamp"),
    before: datetime | None = Query(None, description="Filter activities before this timestamp"),
    per_page: int = Query(30, ge=1, le=200),
    page: int = Query(1, ge=1),
    deps_tuple: tuple[User, Session, Credential] = Depends(deps.require_user),
    db: AsyncSession = Depends(get_session),
) -> list[dict]:
    user, _, credentials = deps_tuple
    activities = await strava.get_activities(
        db,
        credentials,
        after=after,
        before=before,
        per_page=per_page,
        page=page,
    )

    new_cache_entries = 0
    updates = False
    for item in activities:
        strava_id = item.get("id")
        if not strava_id:
            continue
        cache_stmt = select(ActivityCache).where(
            ActivityCache.user_id == user.id,
            ActivityCache.strava_id == strava_id,
        )
        cache_result = await db.exec(cache_stmt)
        cache = cache_result.one_or_none()
        if cache:
            cache.json = json.dumps(item)
            cache.fetched_at = datetime.utcnow()
            updates = True
        else:
            cache = ActivityCache(
                user_id=user.id,
                strava_id=strava_id,
                json=json.dumps(item),
            )
            db.add(cache)
            new_cache_entries += 1

    if new_cache_entries or updates:
        await db.commit()
    return activities
