import copy
import random
import time


class AnswerInserter:

    def __init__(self, answer: str):
        self.answer = answer

    def insert(self, chunks, position_ratio):
        """
        Inserts the answer into one chunk without replacing the document.

        position_ratio:
            0.0 -> beginning
            0.5 -> middle
            1.0 -> end
        """

        start_time = time.time()

        updated_chunks = copy.deepcopy(chunks)

        total_chunks = len(updated_chunks)

        position = round(position_ratio * (total_chunks - 1))

        if position >= total_chunks:
            position = total_chunks - 1

        # Original document
        original_text = updated_chunks[position]["chunk"]

        # Break into sentences
        sentences = [
            s.strip()
            for s in original_text.split(".")
            if s.strip()
        ]

        # Insert answer at a random location inside the document
        insert_index = random.randint(0, len(sentences))

        sentences.insert(
            insert_index,
            self.answer.rstrip(".")
        )

        # Rebuild paragraph
        updated_chunks[position]["chunk"] = ". ".join(sentences) + "."

        insertion_time = time.time() - start_time

        return {
            "chunks": updated_chunks,
            "position_ratio": position_ratio,
            "chunk_number": position + 1,
            "insertion_time": insertion_time
        }
