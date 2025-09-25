# --- ChatGPT summary endpoint ---
import requests
import os, time
import json
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Query
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from stravalib import Client
from dotenv import load_dotenv
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from typing import Optional
from urllib.parse import quote_plus



# load .env values
load_dotenv()

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
SCOPES = os.getenv("SCOPES", "read").split(",")


def to_seconds(value) -> int:
    """
    Convert various 'Duration-like' values to seconds.
    Handles timedelta, objects with .seconds, or numeric values.
    """
    if value is None:
        return 0
    try:
        return int(value.total_seconds())
    except Exception:
        pass
    try:
        return int(getattr(value, "seconds"))
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return 0

app = FastAPI(title="Strava Insights MVP")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'
INDEX_FILE = STATIC_DIR / 'index.html'

app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')

TOKENS_PATH = BASE_DIR / 'tokens.json'


def load_tokens():
    try:
        if TOKENS_PATH.exists():
            data = json.loads(TOKENS_PATH.read_text())
            return {int(k): v for k, v in data.items()}
    except Exception as exc:
        print('[tokens-load-error]', exc)
    return {}


TOKENS = load_tokens()


def save_tokens():
    try:
        payload = {str(k): v for k, v in TOKENS.items()}
        TOKENS_PATH.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        print('[tokens-save-error]', exc)


def normalize_activity_type(value) -> str:
    try:
        if hasattr(value, "value"):
            raw = value.value
        elif hasattr(value, "name"):
            raw = value.name
        else:
            raw = value
    except Exception:
        raw = value

    raw_str = str(raw or "").strip()
    if not raw_str:
        return ""

    lower = raw_str.lower()
    if lower.startswith("activitytype."):
        lower = lower.split(".", 1)[1]
    if lower.startswith("activitytype("):
        lower = lower.split("activitytype(", 1)[1].split(")", 1)[0]
    if "root='" in lower:
        lower = lower.split("root='", 1)[1].split("'", 1)[0]
    return lower.strip()


def build_activity_payload(activity):
    """Serialize a Strava activity into a dict used for AI prompts."""
    return {
        "distance_km": round(float(activity.distance or 0.0) / 1000.0, 2),
        "moving_time_min": round(to_seconds(getattr(activity, "moving_time", None)) / 60, 1),
        "elapsed_time_min": round(to_seconds(getattr(activity, "elapsed_time", None)) / 60, 1),
        "avg_hr": float(getattr(activity, "average_heartrate", 0) or 0),
        "max_hr": float(getattr(activity, "max_heartrate", 0) or 0),
        "elev_gain_m": float(getattr(activity, "total_elevation_gain", 0) or 0),
        "start_date": str(getattr(activity, "start_date", "")),
        "start_latlng": str(getattr(activity, "start_latlng", "")),
        "end_latlng": str(getattr(activity, "end_latlng", "")),
        "pace_min_per_km": round((to_seconds(getattr(activity, "moving_time", None)) / 60) / (float(activity.distance or 0.0) / 1000.0), 2) if getattr(activity, "distance", None) else None,
        "calories": float(getattr(activity, "calories", 0) or 0),
        "type": normalize_activity_type(getattr(activity, "type", "")),
        "name": str(getattr(activity, "name", "")),
        "description": str(getattr(activity, "description", "")),
        "weather": str(getattr(activity, "weather", "")) if hasattr(activity, "weather") else "",
        "location_city": str(getattr(activity, "location_city", "")),
        "location_country": str(getattr(activity, "location_country", "")),
        "achievement_count": int(getattr(activity, "achievement_count", 0)),
        "kudos_count": int(getattr(activity, "kudos_count", 0)),
        "comment_count": int(getattr(activity, "comment_count", 0)),
        "commute": bool(getattr(activity, "commute", False)),
        "private": bool(getattr(activity, "private", False)),
        "visibility": "Private" if getattr(activity, "private", False) else "Public",
        "device_name": str(getattr(activity, "device_name", "")),
        "gear_id": str(getattr(activity, "gear_id", "")),
    }


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RUN_ACTIVITY_TYPES = {"run", "trailrun", "virtualrun"}

# Place this after app = FastAPI(...)

# --- Place below app definition ---

@app.get("/api/activities/{activity_id}/summary")
def activity_summary(activity_id: int):
    if not TOKENS:
        return JSONResponse(status_code=401, content={"error": "Not connected."})
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "OpenAI API key missing."})

    athlete_id = next(iter(TOKENS.keys()))
    client = get_authed_client(athlete_id)
    try:
        activity = client.get_activity(activity_id)
    except Exception as exc:
        print("[summary-load-error]", exc)
        return JSONResponse(status_code=500, content={"error": "Failed to load activity", "detail": str(exc)})

    run_data = build_activity_payload(activity)
    run_data_json = json.dumps(run_data, indent=2)

    prompt = (
        "You are an AI Running Coach. I am a beginner runner who is 182 cm tall and 28 years old. "
        "Please give me a personalized run summary, with actionable advice"
        f"Here is the full run data as a JSON object for reference: {run_data_json}\n"
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system",
                "content": "You are an AI Running Coach. I am a beginner runner who is 182 cm tall and 28 years old."
                "Please give me a personalized run summary, with actionable advice."
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.4,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    max_retries = 3
    backoff = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15,
            )
            if resp.status_code == 429:
                print(f"[summary-error] 429 Too Many Requests, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(backoff ** attempt)
                    continue
                return JSONResponse(status_code=429, content={"error": "OpenAI rate limit exceeded. Please try again later."})
            resp.raise_for_status()
            summary = resp.json()["choices"][0]["message"]["content"]
            return {"summary": summary}
        except Exception as exc:
            last_error = exc
            print(f"[summary-error] {exc}")
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
                continue
            break

    detail = str(last_error) if last_error else "Unknown error"
    return JSONResponse(status_code=500, content={"error": "Failed to generate summary", "detail": detail})

@app.get("/api/activities/{activity_id}/coach")
def activity_coach_plan(activity_id: int):
    if not TOKENS:
        return JSONResponse(status_code=401, content={"error": "Not connected."})
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "OpenAI API key missing."})

    athlete_id = next(iter(TOKENS.keys()))
    client = get_authed_client(athlete_id)
    try:
        activity = client.get_activity(activity_id)
    except Exception as exc:
        print("[coach-plan-activity-error]", exc)
        return JSONResponse(status_code=500, content={"error": "Failed to load activity", "detail": str(exc)})

    run_data = build_activity_payload(activity)
    run_data_json = json.dumps(run_data, indent=2)

    prompt = (
        "You are an AI Running Coach. I am a beginner runner who is 182 cm tall and 28 years old. "
        "Please give me a personalized run summary, with actionable advice "
        f"Here is the recent activity data you should consider: {run_data_json}"
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an encouraging running coach for beginners. Respond in 250 words or less."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 400,
        "temperature": 0.5,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    max_retries = 3
    backoff = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20,
            )
            if resp.status_code == 429:
                print(f"[coach-plan-error] 429 Too Many Requests, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(backoff ** attempt)
                    continue
                return JSONResponse(status_code=429, content={"error": "OpenAI rate limit exceeded. Please try again later."})
            resp.raise_for_status()
            plan = resp.json()["choices"][0]["message"]["content"]
            return {"plan": plan}
        except Exception as exc:
            last_error = exc
            print(f"[coach-plan-error] {exc}")
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
                continue
            break

    detail = str(last_error) if last_error else "Unknown error"
    return JSONResponse(status_code=500, content={"error": "Failed to generate coaching advice", "detail": detail})



@app.get("/api/activities/overall-insight")
def overall_insight(days: int = Query(365, ge=1, le=3650)):
    if not TOKENS:
        return JSONResponse(status_code=401, content={"error": "Not connected."})
    if not OPENAI_API_KEY:
        return JSONResponse(status_code=500, content={"error": "OpenAI API key missing."})

    athlete_id = next(iter(TOKENS.keys()))
    client = get_authed_client(athlete_id)

    after = datetime.utcnow() - timedelta(days=days)
    runs = []
    try:
        for activity in client.get_activities(after=after):
            activity_type = normalize_activity_type(getattr(activity, "type", ""))
            if activity_type not in RUN_ACTIVITY_TYPES:
                continue
            runs.append(build_activity_payload(activity))
    except Exception as exc:
        print('[overall-fetch-error]', exc)
        return JSONResponse(status_code=500, content={"error": "Failed to load activities", "detail": str(exc)})

    if not runs:
        return {"insight": "No runs found in the selected timeframe. Try refreshing your Strava connection."}

    runs_json = json.dumps(runs, indent=2)
    prompt = (
        "Summarise all this running data. Tell me how I am doing for marathon preparation and what I should improve on, "
        "using 250 words or less. Be supportive, highlight strengths, and outline clear next steps."
        f"\n\nHere is the run data as JSON: {runs_json}"
    )

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an experienced marathon coach who gives concise, 250-word maximum insights."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 400,
        "temperature": 0.4,
    }
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

    max_retries = 3
    backoff = 2
    last_error = None

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=20,
            )
            if resp.status_code == 429:
                print(f"[overall-insight-error] 429 Too Many Requests, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    time.sleep(backoff ** attempt)
                    continue
                return JSONResponse(status_code=429, content={"error": "OpenAI rate limit exceeded. Please try again later."})
            resp.raise_for_status()
            insight = resp.json()["choices"][0]["message"]["content"]
            return {"insight": insight}
        except Exception as exc:
            last_error = exc
            print(f"[overall-insight-error] {exc}")
            if attempt < max_retries - 1:
                time.sleep(backoff ** attempt)
                continue
            break

    detail = str(last_error) if last_error else "Unknown error"
    return JSONResponse(status_code=500, content={"error": "Failed to generate overall insight", "detail": detail})

# Serve index.html at root
@app.get("/")
async def serve_frontend():
    return FileResponse(INDEX_FILE)

# allow your React dev server to call the API (optional now, helpful later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_authed_client(athlete_id: int) -> Client:
    t = TOKENS[athlete_id]
    c = Client(access_token=t["access_token"])
    c.refresh_token = t.get("refresh_token")  # keep refresh token for reuse

    if time.time() >= t["expires_at"]:
        refreshed = c.refresh_access_token(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            refresh_token=t["refresh_token"],
        )
        TOKENS[athlete_id].update(refreshed)
        save_tokens()
        c = Client(access_token=refreshed["access_token"])
        c.refresh_token = refreshed.get("refresh_token")
    return c

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/api/strava/login")
def strava_login():
    c = Client()
    url = c.authorization_url(
        client_id=CLIENT_ID,
        redirect_uri=REDIRECT_URI,
        scope=SCOPES,
        approval_prompt="force",  # <= force re-consent
    )
    return RedirectResponse(url)

@app.get("/api/strava/callback")
def strava_callback(code: str = "", error: Optional[str] = None):
    if error:
        return RedirectResponse(url=f"/?auth_error={quote_plus(error)}", status_code=303)
    if not code:
        return RedirectResponse(url="/?auth_error=missing_code", status_code=303)

    try:
        c = Client()
        token = c.exchange_code_for_token(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            code=code,
        )
        c.access_token = token["access_token"]
        athlete = c.get_athlete()
    except Exception as exc:
        print('[strava-callback-error]', exc)
        return RedirectResponse(url="/?auth_error=exchange_failed", status_code=303)

    TOKENS[athlete.id] = {
        "access_token": token["access_token"],
        "refresh_token": token["refresh_token"],
        "expires_at": token["expires_at"],
    }
    save_tokens()

    return RedirectResponse(url="/?connected=1", status_code=303)


@app.get("/api/me/activities")
def my_activities(
    days: int = Query(365, ge=1, le=3650),
    type_filter: Optional[str] = Query(None)
):
    try:
        if not TOKENS:
            return JSONResponse(status_code=401, content={"error": "Not connected. Visit /api/strava/login first"})

        athlete_id = next(iter(TOKENS.keys()))
        c = get_authed_client(athlete_id)

        after = datetime.utcnow() - timedelta(days=days)
        rows = []
        type_counts = {}
        requested_type = type_filter.lower() if type_filter else None

        for a in c.get_activities(after=after):
            try:
                raw_type = getattr(a, "type", None)
                canonical_type = normalize_activity_type(raw_type) or "unknown"

                type_counts[canonical_type] = type_counts.get(canonical_type, 0) + 1

                if requested_type and canonical_type != requested_type:
                    continue

                dist_m = float(a.distance or 0.0)
                distance_km = dist_m / 1000.0
                moving_s = to_seconds(getattr(a, "moving_time", None))
                elapsed_s = to_seconds(getattr(a, "elapsed_time", None))

                pace_spk = (moving_s / distance_km) if (distance_km > 0 and moving_s > 0) else None
                elev_gain_m = float(getattr(a, "total_elevation_gain", 0) or 0)
                avg_hr = float(getattr(a, "average_heartrate", 0) or 0)

                start_dt = getattr(a, "start_date", None)
                try:
                    start_iso = start_dt.isoformat()
                except Exception:
                    start_iso = str(start_dt) if start_dt is not None else None

                rows.append({
                    "id": a.id,
                    "type": canonical_type,
                    "name": a.name or "(no title)",
                    "start_utc": start_iso,
                    "distance_km": round(distance_km, 2),
                    "moving_time_min": round(moving_s / 60, 1) if moving_s else 0,
                    "elapsed_time_min": round(elapsed_s / 60, 1) if elapsed_s else 0,
                    "avg_hr": avg_hr if avg_hr > 0 else None,
                    "elev_gain_m": elev_gain_m if elev_gain_m > 0 else None,
                    "pace_sec_per_km": round(pace_spk, 1) if pace_spk else None,
                })
            except Exception as inner_e:
                print("[activity-parse-error]", repr(inner_e))

        rows.sort(key=lambda r: (r["start_utc"] or ""), reverse=True)
        print(f"[activities-debug] athlete={athlete_id} fetched={len(rows)} type_counts={type_counts}")
        return {"activities": rows}

    except Exception as e:
        print("[activities-endpoint-error]", repr(e))
        return JSONResponse(status_code=500, content={"error": "Failed to load activities", "detail": str(e)})



@app.get("/api/debug/whoami")
def whoami():
    if not TOKENS:
        return {"connected": False}
    athlete_id = next(iter(TOKENS.keys()))
    c = get_authed_client(athlete_id)
    a = c.get_athlete()
    return {"connected": True, "athlete_id": a.id, "name": f"{a.firstname} {a.lastname}"}

@app.get("/api/debug/token")
def token_info():
    if not TOKENS:
        return {"has_token": False}
    athlete_id = next(iter(TOKENS.keys()))
    t = TOKENS[athlete_id].copy()
    # donâ€™t print the full tokens in real apps; just for local debug
    t["access_token"] = "â€¢â€¢â€¢"
    t["refresh_token"] = "â€¢â€¢â€¢"
    return {"has_token": True, "expires_at": t["expires_at"]}


@app.get("/api/insights")
def insights(weeks: int = 8, max_hr: int = 190):
    if not TOKENS:
        return JSONResponse(status_code=401, content={"error": "Not connected. Visit /api/strava/login first"})

    athlete_id = next(iter(TOKENS.keys()))
    c = get_authed_client(athlete_id)

    # pull ~weeks + 2 buffer weeks to be safe
    days = weeks * 7 + 14
    after = datetime.utcnow() - timedelta(days=days)

    from collections import defaultdict
    week_bins = defaultdict(lambda: {"dist_km": 0.0, "time_h": 0.0, "elev_m": 0.0, "count": 0})

    # simple HR zones by % of max
    zones = {"Z1": 0.0, "Z2": 0.0, "Z3": 0.0, "Z4": 0.0, "Z5": 0.0}

    for a in c.get_activities(after=after):
        # only analyze runs for now
        if str(getattr(a, "type", "")).lower() != "run":
            continue

        # weekly bin (ISO year, week number)
        start_dt = getattr(a, "start_date", None)
        if not start_dt:
            continue
        yw = start_dt.isocalendar()[:2]  # (year, week)

        dist_km = float(a.distance or 0.0) / 1000.0
        moving_s = to_seconds(getattr(a, "moving_time", None))
        elev_m = float(getattr(a, "total_elevation_gain", 0) or 0)
        avg_hr = float(getattr(a, "average_heartrate", 0) or 0)

        week_bins[yw]["dist_km"] += dist_km
        week_bins[yw]["time_h"]  += (moving_s / 3600.0)
        week_bins[yw]["elev_m"]  += elev_m
        week_bins[yw]["count"]   += 1

        # HR zones: quick & dirty by % of max
        if avg_hr > 0 and moving_s > 0:
            pct = avg_hr / max_hr
            zone = "Z1" if pct < 0.60 else "Z2" if pct < 0.70 else "Z3" if pct < 0.80 else "Z4" if pct < 0.90 else "Z5"
            zones[zone] += (moving_s / 60.0)  # minutes

    # sort by year/week and keep last N weeks
    items = sorted(week_bins.items())
    last = items[-weeks:]
    trend = [
        {
            "year": yw[0], "week": yw[1],
            "dist_km": round(v["dist_km"], 2),
            "time_h": round(v["time_h"], 2),
            "elev_m": round(v["elev_m"], 0),
            "count": v["count"],
        }
        for (yw, v) in last
    ]
    current = trend[-1] if trend else {"dist_km": 0, "time_h": 0, "elev_m": 0, "count": 0}

    # WoW deltas (vs previous week if available)
    wow = {}
    if len(trend) >= 2:
        prev = trend[-2]
        def delta(a, b):  # current vs prev
            return round(a - b, 2)
        wow = {
            "dist_km": delta(current["dist_km"], prev["dist_km"]),
            "time_h":  delta(current["time_h"],  prev["time_h"]),
            "elev_m":  delta(current["elev_m"],  prev["elev_m"]),
            "count":   current["count"] - prev["count"],
        }

    zones_min = {k: round(v, 1) for k, v in zones.items()}

    return {
        "weekly": trend,
        "currentWeek": current,
        "weekOverWeek": wow,
        "hrZonesMin": zones_min,
        "settings": {"max_hr": max_hr}
    }

@app.get("/api/activities/{activity_id}/streams")
def activity_streams(activity_id: int):
    if not TOKENS:
        return JSONResponse(status_code=401, content={"error": "Not connected."})
    athlete_id = next(iter(TOKENS.keys()))
    c = get_authed_client(athlete_id)

    types = ["time", "heartrate", "velocity_smooth", "distance", "altitude", "cadence"]  # add "latlng" if you want GPS
    s = c.get_activity_streams(activity_id, types=types, resolution="medium")
    # normalize to plain lists
    out = {k: (v.data if hasattr(v, "data") else v) for k, v in s.items()}
    # derive pace (sec/km) if velocity_smooth in m/s and distance in meters are present
    try:
        vel = out.get("velocity_smooth")  # m/s
        dist = out.get("distance")        # m
        if vel and dist:
            # pace = 1000m / (m/s) = sec per km
            out["pace_sec_per_km"] = [round(1000.0 / v, 1) if v and v > 0 else None for v in vel]
    except Exception as e:
        print("[pace-derive-error]", e)
    return out








