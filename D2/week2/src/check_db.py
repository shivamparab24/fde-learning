import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


def list_sessions(cursor: sqlite3.Cursor) -> list[tuple[str, int]]:
    """Fetch unique session IDs and their checkpoint counts."""
    cursor.execute("SELECT thread_id, COUNT(*) FROM checkpoints GROUP BY thread_id;")
    return cursor.fetchall()


def delete_session_records(thread_id: str, db_path: str = "checkpoints.sqlite") -> int:
    """Delete all records associated with a thread_id across all tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 1. Check if the thread_id actually exists in checkpoints
    cursor.execute("SELECT 1 FROM checkpoints WHERE thread_id = ? LIMIT 1;", (thread_id,))
    if not cursor.fetchone():
        print(f"\n[!] Session '{thread_id}' does not exist.")
        conn.close()
        return 0

    # 2. Get all tables and remove matching thread_id rows
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]

    deleted_total = 0
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [col[1] for col in cursor.fetchall()]

        if "thread_id" in columns:
            cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
            deleted_total += cursor.rowcount

    conn.commit()
    conn.close()

    print(f"[✓] Deleted session '{thread_id}' ({deleted_total} database rows removed).")
    return deleted_total


def print_session_summary(checkpointer: SqliteSaver, sessions: list[tuple[str, int]]):
    """Print a clean preview of all active sessions and their latest messages."""
    print("\n" + "=" * 60)
    print(f"ACTIVE SESSIONS SUMMARY ({len(sessions)} total)")
    print("=" * 60)

    for thread_id, count in sessions:
        print(f"\n[Session ID]: {thread_id} ({count} checkpoints)")
        print("-" * 60)

        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = checkpointer.get_tuple(config)

        if checkpoint_tuple:
            messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
            for msg in messages:
                role = msg.__class__.__name__.replace("Message", "").upper()
                content_preview = msg.content.strip().replace("\n", " ")
                if len(content_preview) > 80:
                    content_preview = content_preview[:77] + "..."

                print(f"  {role:<10} | {content_preview}")
        print("-" * 60)


def view_full_session(checkpointer: SqliteSaver, thread_id: str):
    """Print the complete, untruncated message history for a given session."""
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint_tuple = checkpointer.get_tuple(config)

    if not checkpoint_tuple:
        print(f"\n[!] Session '{thread_id}' not found in database.")
        return

    messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
    print("\n" + "=" * 60)
    print(f"FULL HISTORY FOR SESSION: {thread_id}")
    print("=" * 60)

    for msg in messages:
        role = msg.__class__.__name__.replace("Message", "").upper()
        print(f"\n[{role}]:")
        print(msg.content)
        print("-" * 40)


def db_manager(db_path: str = "checkpoints.sqlite"):
    """Main interactive database manager loop."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    while True:
        cursor = conn.cursor()
        sessions = list_sessions(cursor)

        print("\n" + "=" * 60)
        print(f"DATABASE MANAGER: {db_path}")
        print("=" * 60)
        print(" 1. View Sessions Summary")
        print(" 2. Inspect Full Text of a Session")
        print(" 3. Delete a Specific Session")
        print(" 4. Delete ALL Sessions")
        print(" 5. Exit")

        choice = input("\nSelect an action (1-5): ").strip()

        # Action 1: View Summary
        if choice == "1":
            if not sessions:
                print("\n[!] Database is empty. No active sessions found.")
            else:
                print_session_summary(checkpointer, sessions)

        # Action 2: View Full Text
        elif choice == "2":
            if not sessions:
                print("\n[!] No sessions available to inspect.")
                continue

            print("\nAvailable Session IDs:")
            for sid, _ in sessions:
                print(f" - {sid}")

            sid = input("\nEnter Session ID to view: ").strip()
            if sid:
                view_full_session(checkpointer, sid)

        # Action 3: Delete One Session
        elif choice == "3":
            if not sessions:
                print("\n[!] No sessions available to delete.")
                continue

            print("\nAvailable Session IDs:")
            for sid, _ in sessions:
                print(f" - {sid}")

            sid = input("\nEnter Session ID to DELETE: ").strip()
            if sid:
                confirm = input(f"Are you sure you want to delete '{sid}'? (y/N): ").strip().lower()
                if confirm == "y":
                    delete_session_records(sid, db_path)

        # Action 4: Delete ALL Sessions
        elif choice == "4":
            if not sessions:
                print("\n[!] Database is already empty.")
                continue

            confirm = input("\n⚠️ WARNING: Delete ALL sessions? This cannot be undone! (y/N): ").strip().lower()
            if confirm == "y":
                for sid, _ in sessions:
                    delete_session_records(sid, db_path)
                print("\n[✓] Database wiped clean.")

        # Action 5: Exit
        elif choice in ["5", "exit", "quit", "q"]:
            print("\nExiting Database Manager.")
            break

        else:
            print("\n[!] Invalid selection. Please enter a number from 1 to 5.")

    conn.close()


if __name__ == "__main__":
    db_manager()