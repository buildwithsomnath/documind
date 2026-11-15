import logging
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from app.config import Config

logger = logging.getLogger(__name__)

# SQLAlchemy Engine with connection pool recycling and health checks
engine = create_engine(
    Config.DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

# Thread-safe scoped session factory
SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)

# Declarative base for all ORM models
Base = declarative_base()


@contextmanager
def get_db_session():
    """
    Context manager for database transactions.
    Ensures safe commit, automatic rollback on exception, and session cleanup.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db():
    """
    Initializes database schema:
    1. Ensures pgvector extension is created in PostgreSQL.
    2. Registers all models.
    3. Creates tables and indexes if they do not already exist.
    """
    # Import all models so Base registers their metadata before create_all
    import app.models  # noqa: F401

    with engine.connect() as conn:
        logger.info("Verifying pgvector extension in PostgreSQL...")
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    logger.info("Creating database tables if not present...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized successfully.")
