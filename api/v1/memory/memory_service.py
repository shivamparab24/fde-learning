import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from api.v1.schema.schema import CustomerTicketFact, UserProfile
from api.v1.utils.time_utils import utc_now, is_expired, is_stale
from core.config import get_settings
from api.v1.agent.llm import get_llm

settings = get_settings()
logger = logging.getLogger(__name__)

try:
    from langmem import create_memory_store_manager
    LANGMEM_AVAILABLE = True
except ImportError:
    LANGMEM_AVAILABLE = False


EMBEDDING_MODEL_NAME = settings.embedding_model_name
EMBEDDING_DIMS = settings.embedding_dims
MEMORY_NAMESPACE = ("user_memories", "{user_id}")
MEMORY_MAX_AGE_DAYS = int(getattr(settings, "memory_max_age_days", 90))


_embed_fn = None


def _get_embed_fn():
    """Lazy-loaded sentence-transformers embedding function for the vector store index."""
    global _embed_fn
    if _embed_fn is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        def embed(texts: List[str]) -> List[List[float]]:
            return model.encode(list(texts)).tolist()

        _embed_fn = embed
        logger.info("Loaded embedding model '%s' (dims=%d)", EMBEDDING_MODEL_NAME, EMBEDDING_DIMS)
    return _embed_fn


class LangMemManager:
    """
    LangMem long-term memory manager backed by a persistent PostgresStore
    with semantic (vector) search. Falls back to DB-backed memory when
    langmem / pgvector are unavailable.
    """

    def __init__(self):
        self.manager = None
        self.store = None
        self._store_cm = None

        if not LANGMEM_AVAILABLE:
            logger.warning("[LangMem] Package 'langmem' not installed. Falling back to DB memory.")
            return

        try:
            self._init_store()
            llm = get_llm()
            self.manager = create_memory_store_manager(
                llm,
                store=self.store,
                namespace=MEMORY_NAMESPACE,
                schemas=[UserProfile, CustomerTicketFact],
            )
            logger.info("[LangMem] Manager initialized (persistent PostgresStore).")
        except Exception as e:
            logger.error("[LangMem] Initialization error: %s. Falling back to DB memory.", e)
            self.close()
            self.manager = None


    def _init_store(self):
        from langgraph.store.postgres import PostgresStore

        last_error: Optional[Exception] = None
        for with_index in (True, False):
            cm = None
            try:
                kwargs = {}
                if with_index:
                    kwargs["index"] = {
                        "dims": EMBEDDING_DIMS,
                        "embed": _get_embed_fn(),
                    }
                cm = PostgresStore.from_conn_string(settings.database_url, **kwargs)
                store = cm.__enter__()
                store.setup()
                self._store_cm = cm
                self.store = store
                if with_index:
                    logger.info("[LangMem] PostgresStore ready with semantic (vector) search.")
                else:
                    logger.warning(
                        "[LangMem] pgvector unavailable - PostgresStore running without vector search "
                        "(memory retrieval falls back to recency ordering)."
                    )
                return
            except Exception as e:
                last_error = e
                logger.warning("[LangMem] Store init with vector=%s failed: %s", with_index, e)
                if cm is not None:
                    try:
                        cm.__exit__(None, None, None)
                    except Exception:
                        pass

        raise RuntimeError(f"PostgresStore initialization failed: {last_error}")


    def add_memory(
        self,
        person_id: str,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        expires_in_days: Optional[int] = None,
        confidence: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Extracts semantic memories from turn messages using the LangMem manager.
        Memories are persisted in the PostgresStore under ('user_memories', person_id).
        Poisoning safeguards: optional TTL (expires_at) and confidence are stored
        alongside the memory and enforced at retrieval time.
        """
        if not self.manager:
            return []

        meta = dict(metadata or {})
        if expires_in_days is not None:
            meta["expires_at"] = (
                utc_now() + timedelta(days=expires_in_days)
            ).isoformat()
        meta["confidence"] = float(confidence)

        langchain_msgs = []
        for m in messages:
            if m.get("role") == "user":
                langchain_msgs.append(HumanMessage(content=m["content"]))
            elif m.get("role") == "assistant":
                langchain_msgs.append(AIMessage(content=m["content"]))

        try:
            extracted = self.manager.invoke(
                {"messages": langchain_msgs},
                config={"configurable": {"user_id": person_id}},
                metadata=meta,
            )
            return extracted if isinstance(extracted, list) else []
        except Exception as e:
            logger.error("[LangMem] Error processing memory: %s", e)
            return []


    def _is_poisoned(self, value: Dict[str, Any]) -> bool:
        """True when an item is expired or older than the max-age policy."""
        if is_expired(value.get("expires_at")):
            return True
        return is_stale(value.get("created_at"), MEMORY_MAX_AGE_DAYS)

    def search_memories(self, person_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves relevant user memories matching the query via vector similarity.
        Expired / stale memories (poisoning safeguards) are filtered out.
        Returns items formatted consistently with the DB memory fallback.
        """
        if not self.manager or not self.store:
            return []

        try:
            items = self.store.search(("user_memories", person_id), query=query, limit=max(limit * 3, limit))
        except Exception as e:
            logger.error("[LangMem] Error querying memories: %s", e)
            return []

        results = []
        for item in items:
            value = item.value if isinstance(item.value, dict) else {"content": str(item.value)}
            content = value.get("content") or " ".join(str(v) for v in value.values())

            merged = dict(value)
            item_meta = getattr(item, "metadata", None) or {}
            for k in ("expires_at", "confidence"):
                if k in item_meta and k not in merged:
                    merged[k] = item_meta[k]
            if "created_at" not in merged:
                created_attr = getattr(item, "created_at", None)
                if created_attr:
                    merged["created_at"] = created_attr.isoformat() if hasattr(created_attr, "isoformat") else str(created_attr)

            if self._is_poisoned(merged):
                continue

            results.append({
                "content": content,
                "memory_type": value.get("memory_type") or value.get("kind") or "semantic",
                "metadata": {k: v for k, v in merged.items() if k != "content"},
                "created_at": getattr(item, "created_at", None),
                "score": getattr(item, "score", None),
            })
            if len(results) >= limit:
                break
        return results

    def delete_person_memories(self, person_id: str) -> int:
        """
        Deletes all LangMem semantic memories stored under ('user_memories', person_id).
        Returns the number of items removed (0 when unavailable or on error).
        """
        if not self.manager or not self.store:
            return 0

        namespace = ("user_memories", person_id)
        try:
            deleted = 0
            while True:
                items = self.store.search(namespace, limit=100)
                if not items:
                    break
                for item in items:
                    self.store.delete(namespace, item.key)
                    deleted += 1
            logger.info("[LangMem] Deleted %d memories for person '%s'", deleted, person_id)
            return deleted
        except Exception as e:
            logger.error("[LangMem] Error deleting memories for person '%s': %s", person_id, e)
            return 0

    def close(self):
        """Release the PostgresStore connection. Call at app shutdown."""
        if self._store_cm is not None:
            try:
                self._store_cm.__exit__(None, None, None)
            except Exception as e:
                logger.warning("[LangMem] Error closing store: %s", e)
        self._store_cm = None
        self.store = None


_langmem_instance: Optional[LangMemManager] = None


def get_langmem_manager() -> LangMemManager:
    global _langmem_instance
    if _langmem_instance is None:
        _langmem_instance = LangMemManager()
    return _langmem_instance


def shutdown_langmem():
    """Close and discard the singleton manager. Call at app shutdown."""
    global _langmem_instance
    if _langmem_instance is not None:
        _langmem_instance.close()
        _langmem_instance = None
