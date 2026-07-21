class ContextAssembler:
    """
    Builds the prompt for the Lost-in-the-Middle experiment.

    Prompt Structure:

    System Prompt

    Context:
        Document 1
        Document 2
        ...
        Document N

    Question:
        ...
    """

    def assemble(
        self,
        system_prompt: str,
        user_prompt: str,
        retrieved_docs: list[dict]
    ) -> list[dict]:
        context_parts = []
        for doc in retrieved_docs:
            context_parts.append(
                f"Document {doc['id']}\n"
                f"{doc['chunk']}"
            )

        context = "\n\n".join(context_parts)
        user_message = f"""
            Use ONLY the information contained in the documents below.
            If the answer is not present, reply exactly:
            I don't know.
            -------------------- DOCUMENTS --------------------
            {context}
            ---------------------------------------------------
            Question:
            {user_prompt}
            """

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message.strip()
            }
        ]

        return messages
