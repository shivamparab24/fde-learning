from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from core.config import get_settings


def get_engine() -> Engine:
    """Create a SQLAlchemy engine from DATABASE_URL in settings.
    Used exclusively by Alembic — runtime DB access uses the psycopg pool in db.py.
    """
    settings = get_settings()
    # psycopg3 uses 'postgresql+psycopg' driver
    url = settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return create_engine(url)
