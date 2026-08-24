"""Timezone-safe datetime helpers used by poisoning safeguards and TTL policies."""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_iso_utc(value) -> Optional[datetime]:
    """
    Parses an ISO-8601 string/datetime into an aware UTC datetime.
    Naive datetimes are assumed to be UTC. Returns None when unparseable.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            logger.debug("Unparseable datetime value: %r", value)
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def is_expired(expires_at, now: Optional[datetime] = None) -> bool:
    """True when expires_at is set and has passed."""
    exp = parse_iso_utc(expires_at)
    if exp is None:
        return False
    return exp <= (now or utc_now())


def is_stale(created_at, max_age_days: int, now: Optional[datetime] = None) -> bool:
    """True when created_at is older than max_age_days (0 disables the check)."""
    if not created_at or max_age_days <= 0:
        return False
    created = parse_iso_utc(created_at)
    if created is None:
        return False
    cutoff = (now or utc_now()) - timedelta(days=max_age_days)
    return created < cutoff
