"""Database utilities for the ICANRUN Strava dashboard."""
from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./icanrun.db")

_engine: AsyncEngine | None = None
_session_factory: sessionmaker | None = None


def get_engine() -> AsyncEngine:
    """Create (or return) the shared async engine."""
    global _engine, _session_factory
    if _engine is None:
        _engine = create_async_engine(DATABASE_URL, future=True, echo=False)
        _session_factory = sessionmaker(
            _engine, expire_on_commit=False, class_=AsyncSession, autoflush=False
        )
    return _engine


async def init_db() -> None:
    """Initialise database schema if needed."""
    engine = get_engine()
    from . import models  # noqa: F401 ? ensure models are registered

    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session for request-scoped work."""
    if _session_factory is None:
        get_engine()
    assert _session_factory is not None  # for type-checkers
    async with _session_factory() as session:
        yield session
