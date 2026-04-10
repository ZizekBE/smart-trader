"""Async SQLAlchemy engine and session factory.

Both `get_engine` and `get_session_factory` are cached singletons — safe to
call from anywhere without worrying about creating multiple connection pools.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from smart_trader.core.settings import get_settings


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    return create_async_engine(
        get_settings().db_url,
        pool_size=10,
        max_overflow=20,
        echo=False,
    )


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield a session — intended for dependency-injection patterns."""
    async with get_session_factory()() as session:
        yield session
