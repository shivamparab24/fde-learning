# Context Engineering: Best Practices for LLM Context Management

This document explores how to build highly effective Large Language Model (LLM) prompts by managing context size, budgeting tokens, prioritizing critical information, and mitigating the visual and retrieval limits of long-context windows.

---

## 1. Token Management & Budgeting

Efficient context engineering starts with treating the context window as a hard resource constraint. Standard LLMs have strict input token limits, and exceeding them results in truncated requests, lost information, or API errors.

### Token Estimation
Before building a payload, we must estimate token consumption. Standard character or word counts are insufficient due to sub-word tokenization.
* **Tooling:** Libraries like `tiktoken` (for OpenAI models) or model-specific tokenizers (e.g., Llama, Hugging Face tokenizers) are utilized to compute exact token counts programmatically.
* **Cost Allocation:** Every character, space, and metadata wrapper contributes to the total count.

### Token Budgeting
To prevent errors, a structured budget is calculated dynamically before payload assembly:
$$\text{Available Input Tokens} = \text{Model Context Limit} - \text{Reserved Output Tokens} - \text{Safety Buffer}$$

* **Reserved Output Tokens:** Reserving space (e.g., 1,024 or 2,048 tokens) ensures the model doesn't run out of generation capacity mid-response.
* **Safety Buffer:** A small buffer (typically 5-10%) handles minor variations caused by formatting, special tokens, or chat template overhead.

---

## 2. Prompt Assembly & Payload Structure

A robust context assembler takes raw inputs—such as system instructions, user queries, chat history, and retrieved knowledge documents—and packages them into a cleanly formatted payload.

### Payload Structuring
To optimize instruction-following, message histories must be parsed into clear, semantic roles:
1. **System Message (`system`):** Defines the persona, output constraints, and instructions. This is highly static but carries the highest priority.
2. **User Message (`user`):** Includes the active query and any situational context.
3. **Assistant Message (`assistant`):** Contains the historical responses of the LLM.

### Structured Packaging
Within the user or system prompts, documents or context blocks should be isolated using explicit XML-like tags or markdown headers:
```xml
<retrieved_documents>
  <document id="1">
    [Content here]
  </document>
</retrieved_documents>
```
This explicit boundary structure helps the model distinguish instructions from retrieved reference material.

---

## 3. Truncation & Content Reduction Strategies

When the cumulative token count of the history, retrieved documents, and instructions exceeds the allowed token budget, truncation or compression strategies must be applied.

### A. Drop Strategy (First-In, First-Out)
* **Mechanic:** Drops the oldest messages from the conversation history or the lowest-relevance documents first.
* **When to Use:** Suitable for transactional chats where historical continuity from several turns ago is no longer relevant to the active task.

### B. Prioritization & Ranking Strategy
* **Mechanic:** Keeps the most relevant pieces by scoring them against the active user query (e.g., using cosine similarity of embeddings or BM25 lexical scores).
* **When to Use:** Crucial for Retrieval-Augmented Generation (RAG) systems where only the top-$N$ most informative documents should fit into the budget.

### C. Summarization Strategy
* **Mechanic:** Compresses overflowing content rather than dropping it. Older turns of the chat are sent to a lightweight background LLM to produce a concise summary block (e.g., "The user previously asked about database migration steps").
* **When to Use:** Ideal for long, multi-turn conversational agents where preserving high-level context of older interactions is necessary.

---

## 4. Long-Context Phenomena & "Lost in the Middle"

While modern LLMs boast massive context windows (ranging from 32k to over 1M tokens), performance degrades significantly based on *where* information is placed within that window.

### The "Lost in the Middle" Effect
Research has demonstrated that LLMs are highly sensitive to information placement:
* **High Retrieval Accuracy:** Information located at the **very beginning** (primacy effect) or the **very end** (recency effect) of the context window is easily recalled and utilized by the model.
* **Low Retrieval Accuracy:** Information buried in the **middle** of a large context block is frequently missed or ignored by the model's attention heads.

### Mitigation Techniques
1. **Reranking:** Order retrieved documents such that the absolute most important items are placed at the very top or bottom of the context payload, leaving less critical documents in the middle.
2. **Strict Pruning:** Minimize the amount of irrelevant background information passed to the model; more context is not always better.
3. **Chunk Size Optimization:** Keep individual context documents small and highly targeted to maximize retrieval precision.
