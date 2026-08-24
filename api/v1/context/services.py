import json
import logging
from datetime import datetime, timezone
from functools import reduce
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage

from api.v1.context.context_assembler import ContextAssembler
from api.v1.db.db import (
    db_append_chat_turn,
    db_get_chat_history,
    db_get_profile,
    db_get_memories,
    db_add_memory,
    db_count_session_turns,
    db_get_window_messages,
    db_store_summary_block,
    db_get_summary_blocks,
    db_get_messages_after_seq,
)
from api.v1.agent.engine import get_graph, make_thread_id
from api.v1.utils.summarizer import (
    summarize_trimmed_turns,
    summarize_window,
    fold_blocks_into_meta,
)
from api.v1.memory.memory_service import LANGMEM_AVAILABLE, get_langmem_manager
from api.v1.context.prompt import DEFAULT_AMAZON_SYSTEM_PROMPT
from core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# Episodic-compaction knobs (fall back to defaults until present in core/config.py)
SUMMARY_BLOCK_TURNS = int(getattr(settings, "summary_block_turns", 5))
MAX_EPISODE_BLOCKS = int(getattr(settings, "max_episode_blocks", 3))
MEMORY_MAX_AGE_DAYS = int(getattr(settings, "memory_max_age_days", 90))

assembler = ContextAssembler(max_total_tokens=settings.token_budget, reserve_output_tokens=settings.reserve_output_tokens)


def get_profile(person_id: str) -> Dict[str, Any]:
    return db_get_profile(person_id) or {}


def get_relevant_memories(person_id: str, query: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    if LANGMEM_AVAILABLE and query:
        langmem = get_langmem_manager()
        memories = langmem.search_memories(person_id=person_id, query=query, limit=limit)
        if memories:
            return memories
    return db_get_memories(person_id, query=query, limit=limit)


def store_conversation_turn(person_id: str, session_id: str, user_query: str, ai_response: str):
    db_append_chat_turn(person_id, session_id, user_query, ai_response)
    if LANGMEM_AVAILABLE:
        langmem = get_langmem_manager()
        messages = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": ai_response}
        ]
        langmem.add_memory(person_id=person_id, messages=messages, metadata={"session_id": session_id})
    return {"status": "stored"}


def build_context(
    person_id: str,
    session_id: Optional[str],
    system_prompt: Optional[str],
    user_query: str,
    retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    memories: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    profile = get_profile(person_id)
    history = db_get_chat_history(person_id, session_id or "")

    memory_items = memories if memories is not None else get_relevant_memories(person_id, query=user_query)

    return {
        "person_id": person_id,
        "session_id": session_id or "",
        "profile": profile,
        "history": history,
        "memories": memory_items,
        "retrieved_docs": retrieved_docs or [],
        "user_query": user_query,
        "system_prompt": system_prompt.strip() if system_prompt and system_prompt.strip() else DEFAULT_AMAZON_SYSTEM_PROMPT,
    }


def build_prompt(context: Dict[str, Any]) -> str:
    profile = context.get("profile") or {}
    history = context.get("history") or []
    memories = context.get("memories") or []
    retrieved_docs = context.get("retrieved_docs") or []
    user_query = context.get("user_query", "")
    system_prompt = context.get("system_prompt", DEFAULT_AMAZON_SYSTEM_PROMPT)

    profile_block = json.dumps(profile, indent=2, ensure_ascii=False) if profile else "No profile available."
    history_block = "\n".join(
        [f"{item.get('role', 'unknown').upper()}: {item.get('content', '')}" for item in history]
    ) or "No conversation history."
    memory_block = "\n".join(
        [f"- {item.get('content', '')}" for item in memories]
    ) or "No relevant memories."
    docs_block = "\n".join(
        [f"- {doc.get('text', '')}" for doc in retrieved_docs]
    ) or "No retrieved documents."

    return (
        f"{system_prompt}\n\n"
        f"PROFILE:\n{profile_block}\n\n"
        f"RELEVANT MEMORIES:\n{memory_block}\n\n"
        f"CONVERSATION HISTORY:\n{history_block}\n\n"
        f"RETRIEVED DOCS:\n{docs_block}\n\n"
        f"USER QUERY:\n{user_query}"
    )


def build_base_system(context: Dict[str, Any]) -> str:
    """
    Base system payload WITHOUT history/docs/summaries — the assembler owns those
    tiers and injects each exactly once (fixes double-inclusion).
    `build_prompt()` above remains the single-string transparency artifact.
    """
    profile = context.get("profile") or {}
    memories = context.get("memories") or []
    system_prompt = context.get("system_prompt", DEFAULT_AMAZON_SYSTEM_PROMPT)

    profile_block = json.dumps(profile, indent=2, ensure_ascii=False) if profile else "No profile available."
    memory_block = "\n".join(
        [f"- {item.get('content', '')}" for item in memories]
    ) or "No relevant memories."

    return (
        f"{system_prompt}\n\n"
        f"PROFILE:\n{profile_block}\n\n"
        f"RELEVANT MEMORIES:\n{memory_block}"
    )


# ---------------- Episodic compaction helpers ----------------

def _canonical_meta_target(blocks: List[Dict[str, Any]], extra_folds: int = 0) -> Optional[str]:
    """
    The meta-summary must cover all blocks except the most recent K (the ones still
    shown verbatim). Returns None when no folding is required (<=K blocks total).
    `extra_folds` forces one more block into the meta under budget pressure.
    """
    excess = len(blocks) - MAX_EPISODE_BLOCKS - extra_folds
    if excess <= 0:
        return None
    return reduce(fold_blocks_into_meta, [b["content"] for b in blocks[:excess]], "")


def _sync_meta_state(graph, config: dict, blocks: List[Dict[str, Any]]) -> str:
    """Lazily rebuild/persist the meta-summary in the checkpoint's summary channel."""
    target = _canonical_meta_target(blocks)
    current = graph.get_state(config).values.get("summary", "") if graph.get_state(config).values else ""
    if target is not None and target != current:
        graph.update_state(config, {"summary": target})
        logger.info("Meta-summary synced (%d chars) covering folded episode blocks", len(target))
        return target
    return current


def _prune_checkpoint_oldest(graph, config: dict, message_pairs: int):
    """Removes the oldest N turns (2N messages) from checkpoint state after summarization."""
    try:
        state = graph.get_state(config)
        msgs = state.values.get("messages", []) if state and state.values else []
        remove_count = min(message_pairs * 2, len(msgs))
        if remove_count <= 0:
            return
        removals = [RemoveMessage(id=m.id) for m in msgs[:remove_count] if getattr(m, "id", None)]
        if removals:
            graph.update_state(config, {"messages": removals})
            logger.info("Pruned %d summarized messages from checkpoint", len(removals))
    except Exception as e:
        logger.warning("Checkpoint pruning failed: %s", e)


def finalize_window_if_due(person_id: str, session_id: str):
    """
    Post-turn hook: when completed turns hit a multiple of SUMMARY_BLOCK_TURNS,
    generate and persist an immutable episode block, prune those turns from the
    checkpoint, and fold overflow blocks into the running meta-summary.
    """
    try:
        turns = db_count_session_turns(person_id, session_id)
        if turns <= 0 or turns % SUMMARY_BLOCK_TURNS != 0:
            return None

        window_index = turns // SUMMARY_BLOCK_TURNS
        existing = [b for b in db_get_summary_blocks(person_id, session_id) if b["window_index"] == window_index]
        if existing:
            return None  # idempotent: block already stored

        window_msgs, start_seq, end_seq = db_get_window_messages(person_id, session_id, SUMMARY_BLOCK_TURNS)
        content = summarize_window(window_msgs)
        created = db_store_summary_block(person_id, session_id, window_index, content, start_seq, end_seq)
        if not created:
            return None

        logger.info("Episode block #%d stored (%d turns)", window_index, SUMMARY_BLOCK_TURNS)

        thread_id = make_thread_id(person_id, session_id)
        config = {"configurable": {"thread_id": thread_id}}
        graph = get_graph()
        _prune_checkpoint_oldest(graph, config, SUMMARY_BLOCK_TURNS)
        _sync_meta_state(graph, config, db_get_summary_blocks(person_id, session_id))
        return window_index
    except Exception as e:
        logger.error("finalize_window_if_due failed: %s", e)
        return None


def process_chat_turn(
    person_id: str,
    session_id: str,
    system_prompt: str,
    user_query: str,
    retrieved_docs: list,
    memories: Optional[List[Dict[str, Any]]] = None,
    background_tasks: Any = None,
) -> tuple[str, dict, Dict[str, Any], str]:
    """
    Handles episodic-compacted context assembly and executes graph turns.
    Context tiers: base system (persona+profile+memories) -> meta-summary ->
    episode blocks (last K) -> raw recent turns -> RAG docs, each injected once.
    """
    thread_id = make_thread_id(person_id, session_id)
    config = {"configurable": {"thread_id": thread_id}}

    graph = get_graph()

    # --- Episodic state ---
    blocks = db_get_summary_blocks(person_id, session_id)
    last_covered_seq = blocks[-1]["end_seq"] if blocks else None

    current_state = graph.get_state(config)
    raw_history = current_state.values.get("messages", []) if current_state and current_state.values else []
    existing_summary = current_state.values.get("summary", "") if current_state and current_state.values else ""

    if not raw_history:
        # Seed ONLY the unsummarized tail — summarized turns live in episode blocks.
        db_tail = db_get_messages_after_seq(person_id, session_id, last_covered_seq)
        if db_tail:
            formatted_messages = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"])
                for m in db_tail
            ]
            try:
                graph.update_state(config, {"messages": formatted_messages})
                current_state = graph.get_state(config)
                raw_history = current_state.values.get("messages", []) if current_state and current_state.values else []
            except Exception as e:
                logger.warning("Failed to seed checkpoint state from chat_history tail: %s", e)

    history_dicts = []
    for m in raw_history:
        if isinstance(m, HumanMessage):
            history_dicts.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            history_dicts.append({"role": "assistant", "content": m.content})

    context = build_context(
        person_id=person_id,
        session_id=session_id,
        system_prompt=system_prompt,
        user_query=user_query,
        retrieved_docs=retrieved_docs or [],
        memories=memories,
    )

    # Transparency artifact (full single-string prompt) — NOT fed to the model anymore.
    prompt_text = build_prompt(context)

    # Model payload assembles from PARTS: no history/docs duplication.
    base_system = build_base_system(context)

    def _assemble(summary_text: str):
        return assembler.assemble(
            system_prompt=base_system,
            user_query=user_query,
            history=history_dicts,
            retrieved_docs=retrieved_docs or [],
            summary=summary_text,
            episodes=blocks[-MAX_EPISODE_BLOCKS:] if blocks else [],
        )

    assembled = _assemble(existing_summary)

    # Recursive relief under pressure: fold one extra block into the meta-summary and retry.
    if assembled["summary_overflow"] and blocks:
        logger.info("Summary overflow — folding extra episode block into meta-summary")
        target_meta = _canonical_meta_target(blocks, extra_folds=1) or fold_blocks_into_meta(
            existing_summary, blocks[0]["content"]
        )
        existing_summary = target_meta
        assembled = _assemble(existing_summary)

    trimmed_turns = assembled.get("trimmed_history", [])
    updated_summary = existing_summary
    if trimmed_turns:
        # Legacy rolling-summary fallback for un-blocked turns (kept per design).
        updated_summary = summarize_trimmed_turns(existing_summary, trimmed_turns)
        if updated_summary != existing_summary:
            assembled = _assemble(updated_summary)

    assembled_messages = []
    for msg in assembled["messages"]:
        if msg["role"] == "system":
            assembled_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            assembled_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            assembled_messages.append(AIMessage(content=msg["content"]))

    result = graph.invoke(
        {
            "messages": [HumanMessage(content=user_query)],
            "turn_input": assembled_messages,
            "summary": updated_summary,
        },
        config=config,
    )
    latest_msg = result["messages"][-1].content

    telemetry = assembled["telemetry"]
    telemetry["running_summary"] = updated_summary if updated_summary else "(None)"
    telemetry["episode_blocks_total"] = len(blocks)
    telemetry["meta_summary_active"] = bool(existing_summary)

    store_conversation_turn(person_id, session_id, user_query, latest_msg)

    if background_tasks is not None:
        background_tasks.add_task(finalize_window_if_due, person_id, session_id)
    else:
        finalize_window_if_due(person_id, session_id)

    return latest_msg, telemetry, context, prompt_text


def invoke_context_engineering(
    person_id: str,
    session_id: str,
    system_prompt: Optional[str],
    user_query: str,
    retrieved_docs: Optional[list] = None,
    background_tasks: Any = None,
) -> Dict[str, Any]:
    response_text, telemetry, context, prompt = process_chat_turn(
        person_id=person_id,
        session_id=session_id,
        system_prompt=system_prompt or DEFAULT_AMAZON_SYSTEM_PROMPT,
        user_query=user_query,
        retrieved_docs=retrieved_docs or [],
        background_tasks=background_tasks,
    )

    return {
        "person_id": person_id,
        "session_id": session_id,
        "profile": context.get("profile", {}),
        "history": context.get("history", []),
        "memories": context.get("memories", []),
        "context": context,
        "prompt": prompt,
        "response": response_text,
        "telemetry": telemetry,
    }