"""
Database Connection Management

Provides SQLAlchemy engine and session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
import logging

import sys
sys.path.insert(0, '..')
from config.settings import get_settings

logger = logging.getLogger(__name__)


def get_engine():
    """Create SQLAlchemy engine from settings."""
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        pool_size=settings.database_pool_size,
        max_overflow=settings.database_max_overflow,
        echo=settings.debug
    )
    return engine


# Global engine instance
engine = None
SessionLocal = None


def init_db():
    """Initialize database engine and session factory."""
    global engine, SessionLocal
    engine = get_engine()
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Database connection initialized")
    return engine


def get_session() -> Session:
    """Get a new database session."""
    if SessionLocal is None:
        init_db()
    return SessionLocal()


@contextmanager
def get_session_context():
    """Context manager for database sessions with automatic cleanup."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


def create_tables():
    """Create all database tables."""
    from database.models import Base
    if engine is None:
        init_db()
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created")


def drop_tables():
    """Drop all database tables (use with caution)."""
    from database.models import Base
    if engine is None:
        init_db()
    Base.metadata.drop_all(bind=engine)
    logger.warning("Database tables dropped")
