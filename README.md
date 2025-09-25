# ICANRUN Strava Dashboard

Modern FastAPI application that lets each visitor connect their Strava account and view a personal training dashboard. Sessions, Strava tokens, and activity data are stored server-side so multiple users can use the app concurrently.

## Quick start (development)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Create a `.env` file with your keys:

```
STRAVA_CLIENT_ID=12345
STRAVA_CLIENT_SECRET=secret
REDIRECT_URI=https://localhost:8000/api/auth/callback
SCOPES=read,activity:read_all
SESSION_SECRET=replace-me
DATABASE_URL=sqlite+aiosqlite:///./icanrun.db
FRONTEND_ORIGIN=http://localhost:8000
```

Update the Strava developer portal so the redirect URI and domain match your deployment. In production provide a Postgres `DATABASE_URL` (e.g. `postgresql+asyncpg://user:pass@host/db`).

## Project layout

```
app/
  auth.py          # OAuth routes
  db.py            # SQLModel engine and session factory
  deps.py          # Shared dependencies for session/user lookup
  main.py          # FastAPI app wiring
  models.py        # SQLModel tables
  strava.py        # Strava API helpers
  routers/
    activities.py
    me.py
    stats.py
  static/
    landing.html
    dashboard.html
    app.js
```

Acceptance criteria:

- Visiting `/` without a session shows the landing page with a Strava connect button.
- After connecting, `/api/me` and `/api/activities` return the authenticated athlete's data and `401` for anonymous requests.
- Tokens auto-refresh when expired and sessions roll every 14 days.
- `POST /api/auth/logout` clears both cookie and server-side session state.
