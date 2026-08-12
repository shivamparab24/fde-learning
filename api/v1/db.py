import json
import uuid
from typing import List, Dict, Any

import psycopg
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from psycopg_pool import ConnectionPool

from core.config import get_settings
settings = get_settings()

# initialize pool
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


def _ensure_chat_history_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            seq BIGSERIAL PRIMARY KEY,
            id UUID UNIQUE NOT NULL,
            person_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)


def _ensure_profile_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            id UUID PRIMARY KEY,
            person_id TEXT NOT NULL UNIQUE,
            profile_data JSONB NOT NULL DEFAULT '{}'::jsonb,
            updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)


def _ensure_memory_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id UUID PRIMARY KEY,
            person_id TEXT NOT NULL,
            content TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'general',
            metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)


def db_get_all_threads() -> List[str]:
    try:
        with _get_conn() as conn:
            # Ensure required schema elements exist before querying
            _ensure_checkpoint_schema(conn)
            _ensure_chat_history_table(conn)

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

                cursor.execute("""
                    SELECT DISTINCT person_id, session_id
                    FROM chat_history
                    WHERE person_id IS NOT NULL AND session_id IS NOT NULL;
                """)
                for row in cursor.fetchall():
                    threads.add(f"{row['person_id']}::{row['session_id']}")

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

    with _get_conn() as conn:
        _ensure_chat_history_table(conn)

        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM chat_history
                WHERE person_id = %s AND session_id = %s
                """,
                (person_id, session_id),
            )
            deleted_rows += cursor.rowcount

            cursor.execute("""
                SELECT table_name
                FROM information_schema.columns
                WHERE column_name = 'thread_id'
                  AND table_schema = 'public';
            """)
            tables = [row["table_name"] for row in cursor.fetchall()]

            for table in tables:
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

    with _get_conn() as conn:
        _ensure_chat_history_table(conn)
        _ensure_profile_table(conn)
        _ensure_memory_table(conn)

        with conn.cursor() as cursor:
            cursor.execute(
                """
                DELETE FROM chat_history
                WHERE person_id = %s
                """,
                (person_id,),
            )
            deleted_rows += cursor.rowcount

            cursor.execute(
                """
                DELETE FROM memories
                WHERE person_id = %s
                """,
                (person_id,),
            )
            deleted_rows += cursor.rowcount

            cursor.execute(
                """
                DELETE FROM profiles
                WHERE person_id = %s
                """,
                (person_id,),
            )
            deleted_rows += cursor.rowcount

            cursor.execute("""
                SELECT table_name
                FROM information_schema.columns
                WHERE column_name = 'thread_id'
                  AND table_schema = 'public';
            """)
            tables = [row["table_name"] for row in cursor.fetchall()]

            for table in tables:
                query = SQL("DELETE FROM public.{} WHERE thread_id LIKE %s").format(
                    Identifier(table)
                )
                cursor.execute(query, (f"{prefix}%",))
                deleted_rows += cursor.rowcount

        conn.commit()

    return deleted_rows


def db_append_chat_turn(person_id: str, session_id: str, user_query: str, ai_response: str):
    with _get_conn() as conn:
        _ensure_chat_history_table(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat_history (id, person_id, session_id, role, content)
                VALUES (%s, %s, %s, 'user', %s)
                """,
                (str(uuid.uuid4()), person_id, session_id, user_query),
            )
            cursor.execute(
                """
                INSERT INTO chat_history (id, person_id, session_id, role, content)
                VALUES (%s, %s, %s, 'assistant', %s)
                """,
                (str(uuid.uuid4()), person_id, session_id, ai_response),
            )
        conn.commit()


def db_get_chat_history(person_id: str, session_id: str) -> List[Dict[str, str]]:
    with _get_conn() as conn:
        _ensure_chat_history_table(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT role, content
                FROM chat_history
                WHERE person_id = %s AND session_id = %s
                ORDER BY seq ASC;
                """,
                (person_id, session_id),
            )
            rows = cursor.fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


def db_upsert_profile(person_id: str, profile_data: Dict[str, Any]) -> Dict[str, Any]:
    with _get_conn() as conn:
        _ensure_profile_table(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO profiles (id, person_id, profile_data, updated_at)
                VALUES (%s, %s, %s::jsonb, CURRENT_TIMESTAMP)
                ON CONFLICT (person_id)
                DO UPDATE SET profile_data = EXCLUDED.profile_data, updated_at = CURRENT_TIMESTAMP
                RETURNING profile_data;
                """,
                (str(uuid.uuid4()), person_id, json.dumps(profile_data or {})),
            )
            row = cursor.fetchone()
    return row["profile_data"] if row else {}


def db_get_profile(person_id: str) -> Dict[str, Any]:
    with _get_conn() as conn:
        _ensure_profile_table(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT profile_data
                FROM profiles
                WHERE person_id = %s;
                """,
                (person_id,),
            )
            row = cursor.fetchone()
    return row["profile_data"] if row else {}


def db_add_memory(person_id: str, content: str, memory_type: str = "general", metadata: Dict[str, Any] = None):
    with _get_conn() as conn:
        _ensure_memory_table(conn)
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO memories (id, person_id, content, memory_type, metadata)
                VALUES (%s, %s, %s, %s, %s::jsonb);
                """,
                (str(uuid.uuid4()), person_id, content, memory_type, json.dumps(metadata or {})),
            )
        conn.commit()


def db_get_memories(person_id: str, query: str = None, limit: int = 5) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        _ensure_memory_table(conn)
        with conn.cursor() as cursor:
            if query:
                cursor.execute(
                    """
                    SELECT content, memory_type, metadata, created_at
                    FROM memories
                    WHERE person_id = %s
                      AND (
                          content ILIKE %s
                          OR metadata::text ILIKE %s
                      )
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (person_id, f"%{query}%", f"%{query}%", limit),
                )
            else:
                cursor.execute(
                    """
                    SELECT content, memory_type, metadata, created_at
                    FROM memories
                    WHERE person_id = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (person_id, limit),
                )
            rows = cursor.fetchall()

    return [
        {
            "content": row["content"],
            "memory_type": row["memory_type"],
            "metadata": row["metadata"] or {},
            "created_at": row["created_at"],
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