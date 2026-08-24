api/v1/
- __init__.py
- main.py                                       # FastAPI entry point; lifespan wires LangGraph init/close, LangMem store warmup, DB pool shutdown
- index.html                                    # Minimal chat UI served at "/"
- arch.md                                       # This document

- api/router.py                                 # REST endpoints: /invoke + /chat/invoke, profiles, memories, context/prompt build,
                                                #   conversation turns, persons/sessions/history listing, session/person deletes
- schema/schema.py                              # Pydantic request/response models + GraphState
                                                #   channels: messages (append-only raw turns), turn_input (per-call LLM payload), summary

- agent/llm.py                                  # Groq LLM client factory (ChatGroq)
- agent/engine.py                               # LangGraph single-node graph; PostgresSaver checkpointer lifecycle (initialize/close);
                                                #   rolling-summary generation for trimmed turns

- context/prompt.py                             # Default "Amzy" Amazon support system prompt
- context/context_assembler.py                  # Token budgeting (tiktoken): sys+query -> running summary -> windowed history -> RAG docs;
                                                #   emits telemetry + trimmed history
- context/services.py                           # Orchestration: profile/memory retrieval, context & prompt build, chat turn execution,
                                                #   checkpoint seeding from chat_history, turn persistence

- memory/memory_service.py                      # LangMem long-term memory backed by persistent PostgresStore with sentence-transformers
                                                #   vector search ("user_memories" namespace); graceful fallback to Postgres ILIKE search

- db/db.py                                      # psycopg pool (checkpoint introspection, thread deletes) + SQLAlchemy SessionLocal
                                                #   for app tables (chat_history, profiles, memories)
- models/base.py                                # SQLAlchemy DeclarativeBase
- models/models.py                              # ORM models: ChatHistory, Profile, Memory
- models/session.py                             # SQLAlchemy engine factory from settings.database_url (runtime + Alembic)

- migrations/env.py                             # Alembic environment (loads settings, registers models metadata) 
                                                # python -m alembic -c api\v1\migrations\alembic.ini
- migrations/versions/001_initial_schema.py     # Initial schema: chat_history, profiles, memories

Data flow (chat turn):
  POST /chat/invoke or /invoke
    -> services.process_chat_turn
      -> load checkpoint state (seed from chat_history if empty)
      -> build_context (profile + history + memories + docs) -> build_prompt
      -> ContextAssembler.assemble (token budget) ; summarize trimmed turns if any
      -> graph.invoke({turn_input: assembled, messages: [raw user turn], summary})
           llm_node invokes LLM on state["turn_input"], appends AI reply to messages channel
      -> store_conversation_turn (chat_history + LangMem extraction)
