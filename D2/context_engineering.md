# Context Engineering Notes

## Overview

This project explores how to build effective LLM prompts by managing context size, prioritizing important information, and reducing the impact of long-context limitations.

## Main Focus Areas

- Token budgeting
- Prompt assembly
- Conversation history handling
- Retrieved document prioritization
- Truncation and summarization strategies
- Demonstrating the “lost in the middle” effect

## Source Code Overview

### Core source files

- [week_1/src/context_assembler.py](week_1/src/context_assembler.py)  
  Implements a reusable context assembler that reserves output tokens, budgets input tokens, and assembles system instructions, history, and retrieved documents.

- [week_1/src/context_assembly.py](week_1/src/context_assembly.py)  
  Provides a simpler function for assembling message payloads under a strict token budget.

- [week_1/src/pipeline.py](week_1/src/pipeline.py)  
  Shows a practical example of assembling a support-style prompt with metrics.

- [week_1/src/lost_in_middle.py](week_1/src/lost_in_middle.py)  
  Creates synthetic examples to test how context placement affects retrieval performance.

- [week_1/src/demo_lost.py](week_1/src/demo_lost.py)  
  Sends the generated payloads to an Ollama endpoint for live testing.

## Learning Modules

### Prompt and payload structure

- [week_1/learning/context_assembly/assemble_context.py](week_1/learning/context_assembly/assemble_context.py)  
  Demonstrates structured prompt assembly using labeled sections.

- [week_1/learning/payload_structure/payload_structure.py](week_1/learning/payload_structure/payload_structure.py)  
  Shows the structure of a conversation payload with system, user, and assistant messages.

### Token concepts

- [week_1/learning/tokens/token_count.py](week_1/learning/tokens/token_count.py)  
  Uses tiktoken to estimate token usage.

- [week_1/learning/tokens/token_budgeting.py](week_1/learning/tokens/token_budgeting.py)  
  Explains how to reason about token limits and fit content into a budget.

### Truncation strategies

- [week_1/learning/truncation_strategies/drop.py](week_1/learning/truncation_strategies/drop.py)  
  Drops older or lower-priority items when the budget is exceeded.

- [week_1/learning/truncation_strategies/prioritize.py](week_1/learning/truncation_strategies/prioritize.py)  
  Keeps the most relevant pieces by ranking them first.

- [week_1/learning/truncation_strategies/summarise.py](week_1/learning/truncation_strategies/summarise.py)  
  Compresses overflow content into a summary block.

## Key Takeaways

1. A good context builder must respect the token budget.
2. System instructions and the current user query are the highest-priority elements.
3. Conversation history should be trimmed carefully, especially for long chats.
4. Retrieved documents should be ranked and selected based on relevance.
5. Positioning matters: important content placed near the middle may be harder for the model to use effectively.

## Summary

This repository is a practical introduction to context engineering for LLM systems, with examples for budgeting, assembly, prioritization, and retrieval behavior.