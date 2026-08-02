api/v1/
├── __init__.py
├── main.py                # FastAPI entry point & lifespan initialization
├── router.py              # Customer & Support API endpoints
├── schema.py              # Pydantic schemas (ChatRequest, Order, ReturnRequest, Telemetry)
├── db.py                  # PostgreSQL connection & customer/order storage
├── engine.py              # LangGraph graph definition (LLM Node + CS Tools Node)
├── context_assembler.py   # Token budgeting & Lost-in-the-middle context ranking
├── services.py            # Chat orchestration & order service logic
└── llm.py                 # LLM client (Groq / Ollama / OpenAI)
