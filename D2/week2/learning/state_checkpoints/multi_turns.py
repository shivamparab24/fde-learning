import sys
from typing import Annotated
from typing_extensions import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from core.config import get_settings
settings = get_settings()


class State(TypedDict):
    messages: Annotated[list, add_messages]


llm = ChatOllama(
    model=settings.ollama_model_name,
    base_url=settings.ollama_base_url,
    temperature=0.7,
)



def chatbot_node(state: State):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build graph with a checkpoint
builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)





def run_cli_chat(thread_id: str = "cli_session_1"):
    config = {"configurable": {"thread_id": thread_id}}
    
    print("=" * 50)
    print(f"Ollama CLI Chat Initialized (Thread: {thread_id})")
    print("Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            # Handle exit commands
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nEnding session. Goodbye!")
                break

            print("\nBot: ", end="", flush=True)

            # Stream message chunks token-by-token directly to terminal
            input_data = {"messages": [("user", user_input)]}
            for chunk, _ in graph.stream(input_data, config=config, stream_mode="messages"):
                print(chunk.content, end="", flush=True)
            
            print() # New line after message finishes

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession terminated by user. Bye!")
            sys.exit(0)



if __name__ == "__main__":
    run_cli_chat()











# config = {"configurable": {"thread_id": "session_ollama_1"}}

# # Turn 1
# input_1 = {"messages": [("user", "Hello! My name is Jordan and my favorite color is teal.")]}
# res_1 = graph.invoke(input_1, config=config)
# print("Bot:", res_1["messages"][-1].content)

# # Turn 2
# input_2 = {"messages": [("user", "What is my favorite color?")]}
# res_2 = graph.invoke(input_2, config=config)
# print("Bot:", res_2["messages"][-1].content)