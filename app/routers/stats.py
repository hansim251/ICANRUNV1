"""Statistics endpoints built on top of Strava data."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import APIRouter, Depends, Query
from sqlmodel.ext.asyncio.session import AsyncSession

from .. import deps, strava
from ..db import get_session
from ..models import Credential, Session, User

router = APIRouter(prefix="/api", tags=["stats"])


def _parse_start_date(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value.replace("Z", "+00:00")
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@router.get("/stats")
async def get_stats(
    days: int = Query(90, ge=7, le=365),
    deps_tuple: tuple[User, Session, Credential] = Depends(deps.require_user),
    db: AsyncSession = Depends(get_session),
) -> Dict[str, object]:
    _, _, credentials = deps_tuple
    after = datetime.utcnow() - timedelta(days=days)

    per_page = 100
    page = 1
    activities: List[dict] = []
    while page <= 5:  # cap pages to keep latency reasonable
        batch = await strava.get_activities(
            db,
            credentials,
            after=after,
            per_page=per_page,
            page=page,
        )
        if not batch:
            break
        activities.extend(batch)
        if len(batch) < per_page:
            break
        page += 1

    total_distance_m = 0.0
    total_moving_time_s = 0.0
    total_elev_gain = 0.0
    hr_sum = 0.0
    hr_samples = 0

    weekly: Dict[str, Dict[str, float | int]] = {}

    for activity in activities:
        distance = float(activity.get("distance") or 0.0)
        moving_time = float(activity.get("moving_time") or 0.0)
        elev_gain = float(activity.get("total_elevation_gain") or 0.0)
        avg_hr = activity.get("average_heartrate")

        total_distance_m += distance
        total_moving_time_s += moving_time
        total_elev_gain += elev_gain

        if avg_hr:
            hr_sum += float(avg_hr) * moving_time
            hr_samples += int(moving_time)

        start_at = _parse_start_date(activity.get("start_date"))
        if not start_at:
            continue
        iso_year, iso_week, _ = start_at.isocalendar()
        key = f"{iso_year}-W{iso_week:02d}"
        bucket = weekly.setdefault(
            key,
            {"distance_km": 0.0, "time_hours": 0.0, "elev_gain_m": 0.0, "count": 0},
        )
        bucket["distance_km"] += distance / 1000.0
        bucket["time_hours"] += moving_time / 3600.0
        bucket["elev_gain_m"] += elev_gain
        bucket["count"] += 1

    avg_pace_sec_per_km = None
    if total_distance_m > 0 and total_moving_time_s > 0:
        avg_pace_sec_per_km = total_moving_time_s / (total_distance_m / 1000.0)

    avg_hr = None
    if hr_samples > 0:
        avg_hr = hr_sum / hr_samples

    return {
        "sample_size": len(activities),
        "window_days": days,
        "totals": {
            "distance_km": round(total_distance_m / 1000.0, 2),
            "moving_time_hours": round(total_moving_time_s / 3600.0, 2),
            "elev_gain_m": round(total_elev_gain, 1),
            "avg_hr": round(avg_hr, 1) if avg_hr else None,
            "avg_pace_sec_per_km": round(avg_pace_sec_per_km, 1) if avg_pace_sec_per_km else None,
        },
        "weekly": [
            {
                "week": key,
                "distance_km": round(values["distance_km"], 2),
                "time_hours": round(values["time_hours"], 2),
                "elev_gain_m": round(values["elev_gain_m"], 1),
                "count": values["count"],
            }
            for key, values in sorted(weekly.items())
        ],
    }
