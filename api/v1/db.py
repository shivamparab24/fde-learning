import os
from typing import List, Dict, Any
import uuid

import psycopg
from psycopg.rows import dict_row

from core.config import get_settings
settings = get_settings()


def _get_conn():
    conn_params = {
        "host": settings.postgres_host,
        "port": int(settings.postgres_port),
        "dbname": settings.postgres_db,
        "user": settings.postgres_user,
        "password": settings.postgres_password,
        "row_factory": dict_row,
    }
    return psycopg.connect(**conn_params, autocommit=True)


def _ensure_chat_history_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id UUID PRIMARY KEY,
            person_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()

def db_get_all_threads() -> List[str]:
    try:
        with _get_conn() as conn:
            _ensure_chat_history_table(conn)

            with conn.cursor() as cursor:
                threads = set()

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
    """
    Delete rows where thread_id matches.
    - exact=True => exact match on thread_id
    - prefix=True => thread_id LIKE 'pattern%'
    - default => uses LIKE '%pattern%' (legacy behavior)
    """
    deleted_rows = 0
    with _get_conn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT table_name
                FROM information_schema.columns
                WHERE column_name = 'thread_id'
                  AND table_schema = 'public';
            """)
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                if exact:
                    cursor.execute(
                        f'DELETE FROM public."{table}" WHERE thread_id = %s',
                        (thread_id_pattern,),
                    )
                elif prefix:
                    cursor.execute(
                        f'DELETE FROM public."{table}" WHERE thread_id LIKE %s',
                        (f"{thread_id_pattern}%",),
                    )
                else:
                    cursor.execute(
                        f'DELETE FROM public."{table}" WHERE thread_id LIKE %s',
                        (f"%{thread_id_pattern}%",),
                    )
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
                ORDER BY timestamp ASC, id ASC;
                """,
                (person_id, session_id),
            )
            rows = cursor.fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


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