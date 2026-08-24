import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional, Tuple
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from psycopg_pool import ConnectionPool

from sqlalchemy import cast, String, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import sessionmaker
from api.v1.models.session import get_engine
from api.v1.models.models import ChatHistory, Person, Profile, Memory, SessionSummary

from core.config import get_settings
settings = get_settings()
logger = logging.getLogger(__name__)

# initialize psycopg pool for LangGraph / checkpoint introspection
conn_params = {
    "host": settings.postgres_host,
    "port": int(settings.postgres_port),
    "dbname": settings.postgres_db,
    "user": settings.postgres_user,
    "password": settings.postgres_password,
    "row_factory": dict_row,
}

pool = ConnectionPool(
    min_size=4,
    max_size=20,
    kwargs=conn_params,
    open=True,
)

def _get_conn():
    return pool.connection()


# initialize SQLAlchemy sessionmaker
engine = get_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _ensure_person(session, person_id: str):
    """Idempotently anchor a person row (parent of all FK-cascade children) in the caller's transaction."""
    stmt = pg_insert(Person).values(person_id=person_id).on_conflict_do_nothing(
        index_elements=[Person.person_id]
    )
    session.execute(stmt)


def db_get_all_threads() -> List[str]:
    try:
        with _get_conn() as conn:
            threads = set()

            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT DISTINCT thread_id
                    FROM checkpoints
                    WHERE thread_id IS NOT NULL;
                """)
                for row in cursor.fetchall():
                    if row["thread_id"]:
                        threads.add(row["thread_id"])

        with get_session() as session:
            rows = session.query(ChatHistory.person_id, ChatHistory.session_id).distinct().all()
            for r in rows:
                if r.person_id and r.session_id:
                    threads.add(f"{r.person_id}::{r.session_id}")

        return sorted(threads)
    except Exception as e:
        logger.warning("db_get_all_threads failed: %s", e)
        return []


def db_delete_thread(thread_id_pattern: str, prefix: bool = False, exact: bool = False) -> int:
    deleted_rows = 0
    with _get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name
                FROM information_schema.columns
                WHERE column_name = 'thread_id'
                  AND table_schema = 'public';
            """)
            tables = [row["table_name"] for row in cursor.fetchall()]

            for table in tables:
                if exact:
                    operator = SQL("=")
                    pattern = thread_id_pattern
                elif prefix:
                    operator = SQL("LIKE")
                    pattern = f"{thread_id_pattern}%"
                else:
                    operator = SQL("LIKE")
                    pattern = f"%{thread_id_pattern}%"

                query = SQL("DELETE FROM public.{} WHERE thread_id {} %s").format(
                    Identifier(table),
                    operator
                )

                cursor.execute(query, (pattern,))
                deleted_rows += cursor.rowcount
        conn.commit()
    return deleted_rows


def db_delete_session(person_id: str, session_id: str) -> int:
    deleted_rows = 0
    thread_id = f"{person_id}::{session_id}"

    with get_session() as session:
        res = session.query(ChatHistory).filter(
            ChatHistory.person_id == person_id,
            ChatHistory.session_id == session_id
        ).delete(synchronize_session=False)
        deleted_rows += res

        res_sum = session.query(SessionSummary).filter(
            SessionSummary.person_id == person_id,
            SessionSummary.session_id == session_id
        ).delete(synchronize_session=False)
        deleted_rows += res_sum

    with _get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name
                FROM information_schema.columns
                WHERE column_name = 'thread_id'
                  AND table_schema = 'public';
            """)
            tables = [row["table_name"] for row in cursor.fetchall()]

            for table in tables:
                if table != "chat_history":
                    query = SQL("DELETE FROM public.{} WHERE thread_id = %s").format(
                        Identifier(table)
                    )
                    cursor.execute(query, (thread_id,))
                    deleted_rows += cursor.rowcount

        conn.commit()

    return deleted_rows


def db_delete_person(person_id: str) -> int:
    deleted_rows = 0
    prefix = f"{person_id}::"

    with get_session() as session:
        res_chat = session.query(ChatHistory).filter(ChatHistory.person_id == person_id).delete(synchronize_session=False)
        res_mem = session.query(Memory).filter(Memory.person_id == person_id).delete(synchronize_session=False)
        res_prof = session.query(Profile).filter(Profile.person_id == person_id).delete(synchronize_session=False)
        res_pers = session.query(Person).filter(Person.person_id == person_id).delete(synchronize_session=False)
        deleted_rows += (res_chat + res_mem + res_prof + res_pers)

    with _get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name
                FROM information_schema.columns
                WHERE column_name = 'thread_id'
                  AND table_schema = 'public';
            """)
            tables = [row["table_name"] for row in cursor.fetchall()]

            for table in tables:
                if table != "chat_history":
                    query = SQL("DELETE FROM public.{} WHERE thread_id LIKE %s").format(
                        Identifier(table)
                    )
                    cursor.execute(query, (f"{prefix}%",))
                    deleted_rows += cursor.rowcount

        conn.commit()

    return deleted_rows


def db_append_chat_turn(person_id: str, session_id: str, user_query: str, ai_response: str):
    with get_session() as session:
        _ensure_person(session, person_id)
        user_turn = ChatHistory(
            id=uuid.uuid4(),
            person_id=person_id,
            session_id=session_id,
            role="user",
            content=user_query
        )
        ai_turn = ChatHistory(
            id=uuid.uuid4(),
            person_id=person_id,
            session_id=session_id,
            role="assistant",
            content=ai_response
        )
        session.add(user_turn)
        session.add(ai_turn)


def db_get_chat_history(person_id: str, session_id: str) -> List[Dict[str, str]]:
    with get_session() as session:
        rows = session.query(ChatHistory).filter(
            ChatHistory.person_id == person_id,
            ChatHistory.session_id == session_id
        ).order_by(ChatHistory.seq.asc()).all()
    return [{"role": row.role, "content": row.content} for row in rows]


def db_upsert_profile(person_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    with get_session() as session:
        _ensure_person(session, person_id)
        profile = session.query(Profile).filter(Profile.person_id == person_id).first()
        if profile:
            profile.profile_data = profile_data or {}
        else:
            profile = Profile(
                id=uuid.uuid4(),
                person_id=person_id,
                profile_data=profile_data or {}
            )
            session.add(profile)
        session.flush()
        res_data = profile.profile_data
    return res_data


def db_get_profile(person_id: str) -> Dict[str, Any]:
    with get_session() as session:
        profile = session.query(Profile).filter(Profile.person_id == person_id).first()
    return profile.profile_data if profile else {}


def db_add_memory(
    person_id: str,
    content: str,
    memory_type: str = "general",
    metadata: Dict[str, Any] = None,
    expires_at: Optional[datetime] = None,
    confidence: float = 1.0,
):
    with get_session() as session:
        _ensure_person(session, person_id)
        memory = Memory(
            id=uuid.uuid4(),
            person_id=person_id,
            content=content,
            memory_type=memory_type,
            metadata_=metadata or {},
            expires_at=expires_at,
            confidence=confidence,
        )
        session.add(memory)


def db_get_memories(person_id: str, query: str = None, limit: int = 5, max_age_days: int = None) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc)
    with get_session() as session:
        q = session.query(Memory).filter(Memory.person_id == person_id)
        # Poisoning safeguard: never surface expired memories.
        q = q.filter((Memory.expires_at.is_(None)) | (Memory.expires_at > now))
        if query:
            search = f"%{query}%"
            q = q.filter(
                (Memory.content.ilike(search)) |
                (cast(Memory.metadata_, String).ilike(search))
            )
        if max_age_days is not None:
            cutoff = now - timedelta(days=max_age_days)
            q = q.filter(Memory.created_at >= cutoff)
        rows = q.order_by(Memory.created_at.desc()).limit(limit).all()

    return [
        {
            "content": row.content,
            "memory_type": row.memory_type,
            "metadata": row.metadata_ or {},
            "created_at": row.created_at,
            "expires_at": row.expires_at,
            "confidence": float(row.confidence) if row.confidence is not None else 1.0,
        }
        for row in rows
    ]


def db_purge_expired_memories(person_id: str = None) -> int:
    """Deletes memories whose expires_at has passed. Optionally scoped to one person."""
    now = datetime.now(timezone.utc)
    with get_session() as session:
        q = session.query(Memory).filter(Memory.expires_at.isnot(None), Memory.expires_at <= now)
        if person_id:
            q = q.filter(Memory.person_id == person_id)
        return q.delete(synchronize_session=False)


# ---------------- Episodic summary blocks ----------------

def db_count_session_turns(person_id: str, session_id: str) -> int:
    """A 'turn' is one user message (assistant reply is part of the same turn)."""
    with get_session() as session:
        count = session.query(func.count(ChatHistory.seq)).filter(
            ChatHistory.person_id == person_id,
            ChatHistory.session_id == session_id,
            ChatHistory.role == "user",
        ).scalar()
    return int(count or 0)


def db_get_window_messages(person_id: str, session_id: str, num_turns: int) -> Tuple[List[Dict[str, str]], Optional[int], Optional[int]]:
    """
    Returns (messages, start_seq, end_seq) for the most recent `num_turns` turns.
    Messages are ordered oldest-first with role/content keys; seq bounds span the window.
    """
    fetch = max(1, num_turns) * 2
    with get_session() as session:
        rows = (
            session.query(ChatHistory)
            .filter(ChatHistory.person_id == person_id, ChatHistory.session_id == session_id)
            .order_by(ChatHistory.seq.desc())
            .limit(fetch)
            .all()
        )
    rows = list(reversed(rows))
    if not rows:
        return [], None, None
    messages = [{"role": r.role, "content": r.content} for r in rows]
    return messages, rows[0].seq, rows[-1].seq


def db_store_summary_block(
    person_id: str,
    session_id: str,
    window_index: int,
    content: str,
    start_seq: Optional[int] = None,
    end_seq: Optional[int] = None,
) -> bool:
    """Idempotent insert of an immutable episode block. Returns True when a new row was created."""
    if not content.strip():
        return False
    with get_session() as session:
        _ensure_person(session, person_id)
        stmt = (
            pg_insert(SessionSummary)
            .values(
                id=uuid.uuid4(),
                person_id=person_id,
                session_id=session_id,
                window_index=window_index,
                start_seq=start_seq,
                end_seq=end_seq,
                content=content.strip(),
            )
            .on_conflict_do_nothing(constraint="uq_session_summary_window")
        )
        result = session.execute(stmt)
        return bool(result.rowcount)


def db_get_summary_blocks(person_id: str, session_id: str) -> List[Dict[str, Any]]:
    """All episode blocks for the session, oldest-first."""
    with get_session() as session:
        rows = (
            session.query(SessionSummary)
            .filter(SessionSummary.person_id == person_id, SessionSummary.session_id == session_id)
            .order_by(SessionSummary.window_index.asc())
            .all()
        )
    return [
        {
            "window_index": r.window_index,
            "start_seq": r.start_seq,
            "end_seq": r.end_seq,
            "content": r.content,
            "created_at": r.created_at,
        }
        for r in rows
    ]


def db_get_messages_after_seq(person_id: str, session_id: str, after_seq: Optional[int]) -> List[Dict[str, str]]:
    """Raw conversation tail that is not yet covered by any episode block."""
    with get_session() as session:
        q = session.query(ChatHistory).filter(
            ChatHistory.person_id == person_id,
            ChatHistory.session_id == session_id,
        )
        if after_seq is not None:
            q = q.filter(ChatHistory.seq > after_seq)
        rows = q.order_by(ChatHistory.seq.asc()).all()
    return [{"role": r.role, "content": r.content} for r in rows]
