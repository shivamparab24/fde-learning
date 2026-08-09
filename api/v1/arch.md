api/v1/
- __init__.py
- main.py                    # FastAPI entry point and app lifespan setup
- router.py                  # API routes for chat, history, profiles, memories, context, prompts, and conversation turns
- schema.py                  # Pydantic models for chat requests/responses and profile/memory/context/prompt payloads
- db.py                      # PostgreSQL access for profiles, memories, chat history, and conversation storage
- engine.py                  # LangGraph graph definition and summary generation logic
- context_assembler.py       # Token budgeting and context assembly from history, memories, and retrieved docs
- services.py                # Orchestration for profile lookup, memory retrieval, context building, prompt building, and chat processing
- llm.py                     # LLM client configuration and initialization