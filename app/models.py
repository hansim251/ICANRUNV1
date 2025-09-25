"""SQLModel models for users, sessions, and Strava data."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel

SESSION_TTL_DAYS = 14


def default_expires() -> datetime:
    """Return the default session expiry timestamp."""
    return datetime.utcnow() + timedelta(days=SESSION_TTL_DAYS)


class User(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid4, primary_key=True)
    athlete_id: int = Field(index=True, unique=True, nullable=False)
    username: Optional[str] = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

    credentials: Credential | None = Relationship(back_populates="user", sa_relationship_kwargs={"uselist": False})
    sessions: list[Session] = Relationship(back_populates="user")


class Session(SQLModel, table=True):
    id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True, index=True)
    user_id: Optional[UUID] = Field(default=None, foreign_key="user.id")
    state: Optional[str] = Field(default=None, index=True)
    pending_redirect: Optional[str] = Field(default=None, nullable=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    last_seen: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    expires_at: datetime = Field(default_factory=default_expires, nullable=False)

    user: User | None = Relationship(back_populates="sessions")


class Credential(SQLModel, table=True):
    user_id: UUID = Field(foreign_key="user.id", primary_key=True)
    access_token: str = Field(nullable=False)
    refresh_token: str = Field(nullable=False)
    expires_at: datetime = Field(nullable=False)
    scope: Optional[str] = Field(default=None, nullable=True)

    user: User | None = Relationship(back_populates="credentials")


class ActivityCache(SQLModel, table=True):
    __table_args__ = (UniqueConstraint("user_id", "strava_id", name="uq_activity_user_strava"),)

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: UUID = Field(foreign_key="user.id", nullable=False, index=True)
    strava_id: int = Field(nullable=False)
    json: str = Field(nullable=False)
    fetched_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)

