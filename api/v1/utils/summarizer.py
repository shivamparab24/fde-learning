import logging
from langchain_core.messages import HumanMessage

from api.v1.agent.llm import get_llm

logger = logging.getLogger(__name__)

llm = get_llm()


def _llm_summarize(prompt: str) -> str:
    """Shared LLM summarization call with error handling. Returns '' on failure."""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        summary_text = response.content.strip()
        logger.info("LLM summary generated: %d chars", len(summary_text))
        return summary_text
    except Exception as e:
        logger.error("Error generating LLM summary: %s", e)
        return ""


def _format_turns(turns: list) -> str:
    return "\n".join(
        [f"{msg['role'].upper()}: {msg['content']}" for msg in turns]
    )


def summarize_trimmed_turns(existing_summary: str, trimmed_turns: list) -> str:
    """Summarizes older/trimmed conversation turns into a single updated summary string."""

    logger.info("Generating rolling summary from %d trimmed turns", len(trimmed_turns))
    if not trimmed_turns:
        return existing_summary

    formatted_turns = _format_turns(trimmed_turns)

    if existing_summary.strip():
        prompt = (
            f"Existing summary of earlier conversation:\n{existing_summary}\n\n"
            f"New conversation turns to integrate:\n{formatted_turns}\n\n"
            "Write a single concise summary paragraph incorporating all key facts, topics, preferences, and decisions."
        )
    else:
        prompt = (
            f"Summarize key facts, topics, decisions, and user details from these conversation turns into a concise summary paragraph:\n{formatted_turns}"
        )

    result = _llm_summarize(prompt)
    return result if result else existing_summary


def summarize_window(window_turns: list) -> str:
    """
    Summarizes one fixed window of conversation turns (an 'episode block').
    Unlike the rolling summary, this is immutable once created: it describes
    exactly the turns in this window and is never merged with other content.
    Returns '' on failure (caller should skip storing an empty block).
    """
    logger.info("Generating episode summary for %d turns", len(window_turns))
    if not window_turns:
        return ""

    prompt = (
        "Summarize this short exchange window into a compact episode block. "
        "Capture only durable information: user intent, facts stated, decisions made, "
        "outcomes and any open follow-ups. Maximum 120 words.\n\n"
        f"{_format_turns(window_turns)}"
    )
    return _llm_summarize(prompt)


def fold_blocks_into_meta(existing_meta: str, oldest_block: str) -> str:
    """
    Recursively compacts memory of past episodes: merges the OLDEST episode
    block into the running meta-summary. This is summary-of-summaries —
    the meta-summary never contains raw turns, only distilled episode knowledge.
    Returns prior meta unchanged on failure.
    """
    if not (oldest_block or "").strip():
        return existing_meta

    logger.info("Folding oldest episode block into running meta-summary")
    if (existing_meta or "").strip():
        prompt = (
            f"Running summary of older episodes:\n{existing_meta}\n\n"
            f"Episode to merge into it:\n{oldest_block}\n\n"
            "Write a single updated compact summary that preserves every still-relevant fact, "
            "decision, preference or open item. Drop anything superseded by the newer episode. "
            "Maximum 150 words."
        )
    else:
        prompt = (
            "Summarize this episode into a compact running summary preserving all durable "
            "facts, preferences, decisions and open items. Maximum 150 words.\n\n"
            f"{oldest_block}"
        )
    result = _llm_summarize(prompt)
    return result if result else existing_meta
