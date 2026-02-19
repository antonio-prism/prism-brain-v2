"""
PRISM Brain Database Connection
PostgreSQL connection management using SQLAlchemy.
"""

import os
import logging
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base

logger = logging.getLogger(__name__)

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Railway uses postgres:// but SQLAlchemy needs postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine with connection pooling
engine = None
SessionLocal = None


def init_db():
    """Initialize database engine and create tables."""
    global engine, SessionLocal

    if not DATABASE_URL:
        logger.error("DATABASE_URL not set!")
        raise ValueError("DATABASE_URL environment variable is required")

    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=1800,
        pool_pre_ping=True,
        echo=False
    )

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully")

    return engine


@contextmanager
def get_session_context():
    """Context manager for database sessions."""
    if SessionLocal is None:
        init_db()

    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Session:
    """Dependency for FastAPI endpoints."""
    if SessionLocal is None:
        init_db()

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
