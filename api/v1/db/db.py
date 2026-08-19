import uuid
from typing import List, Dict, Any
from contextlib import contextmanager

from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from psycopg_pool import ConnectionPool

from sqlalchemy import cast, String
from sqlalchemy.orm import sessionmaker
from api.v1.models.session import get_engine
from api.v1.models.models import ChatHistory, Profile, Memory

from core.config import get_settings
settings = get_settings()

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


def db_get_all_threads() -> List[str]:
    try:
        with _get_conn() as conn:
            # Ensure LangGraph checkpoint schema is up-to-date
            _ensure_checkpoint_schema(conn)

            with conn.cursor() as cursor:
                threads = set()

                # Guard check: query checkpoints only if table actually exists
                cursor.execute("""
                    SELECT 1 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' AND table_name = 'checkpoints';
                """)
                if cursor.fetchone():
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
    except Exception:
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
        deleted_rows += (res_chat + res_mem + res_prof)

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


def db_add_memory(person_id: str, content: str, memory_type: str = "general", metadata: Dict[str, Any] = None):
    with get_session() as session:
        memory = Memory(
            id=uuid.uuid4(),
            person_id=person_id,
            content=content,
            memory_type=memory_type,
            metadata_=metadata or {}
        )
        session.add(memory)


def db_get_memories(person_id: str, query: str = None, limit: int = 5) -> List[Dict[str, Any]]:
    with get_session() as session:
        q = session.query(Memory).filter(Memory.person_id == person_id)
        if query:
            search = f"%{query}%"
            q = q.filter(
                (Memory.content.ilike(search)) |
                (cast(Memory.metadata_, String).ilike(search))
            )
        rows = q.order_by(Memory.created_at.desc()).limit(limit).all()

    return [
        {
            "content": row.content,
            "memory_type": row.memory_type,
            "metadata": row.metadata_ or {},
            "created_at": row.created_at,
        }
        for row in rows
    ]


def _ensure_checkpoint_schema(conn):
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = 'checkpoint_blobs';
        """)
        blobs_exists = cursor.fetchone()
        if blobs_exists:
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'checkpoint_blobs' AND column_name = 'version';
            """)
            has_version = cursor.fetchone()
            if not has_version:
                cursor.execute("DROP TABLE IF EXISTS checkpoint_blobs CASCADE;")
                cursor.execute("DROP TABLE IF EXISTS checkpoints CASCADE;")
                cursor.execute("DROP TABLE IF EXISTS checkpoint_writes CASCADE;")
                cursor.execute("DROP TABLE IF EXISTS checkpoint_migrations CASCADE;")
    conn.commit()
    