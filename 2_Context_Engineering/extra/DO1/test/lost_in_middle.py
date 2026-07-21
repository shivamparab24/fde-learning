from DO1.core.answer_inserter import AnswerInserter
from DO1.data.generate_dummy_data import DummyDataGenerator
# from context.answer_inserter import AnswerInserter
from DO1.context.context_assembler import ContextAssembler
from DO1.core.llm import OllamaLLM


class LostInMiddleExperiment:

    def __init__(self, system_prompt, query, answer):
        self.system_prompt = system_prompt
        self.query = query
        self.answer = answer

        self.generator = DummyDataGenerator(total_chunks=500)
        self.context_assembler = ContextAssembler()
        self.llm = OllamaLLM()

        # 5 positions = 5 separate contexts created
        self.positions = [0.0, 0.25, 0.5, 0.75, 1.0]

    def evaluate(self, response):
        response_clean = response.lower()
        refusals = ["don't know", "do not know", "not specified", "cannot answer"]
        if any(r in response_clean for r in refusals):
            return False
        return "12" in response_clean and "sick leave" in response_clean

    def run(self):
        print(f"\nRunning Experiment across {len(self.positions)} unique contexts...\n")

        results = []

        # Iterate through each position and generate a dedicated context
        for idx, position in enumerate(self.positions, start=1):
            print("=" * 70)
            print(f"Context #{idx} | Position Ratio: {position:.2f}")
            print("=" * 70)

            # Generate a specific context where the needle is placed at 'position'
            chunks = self.generator.generate(position=position)

            # Assemble prompt using the generated context chunks
            messages = self.context_assembler.assemble(
                system_prompt=self.system_prompt,
                user_prompt=self.query,
                retrieved_docs=chunks
            )

            # Query model
            response = self.llm.generate(messages)
            prediction = response["response"]
            correct = self.evaluate(prediction)

            # Target chunk index (1-indexed for display)
            target_chunk = min(int(position * self.generator.total_chunks), self.generator.total_chunks - 1) + 1

            result = {
                "context_id": idx,
                "position": position,
                "chunk_number": target_chunk,
                "correct": correct,
                "processing_time": response["processing_time"],
                "prediction": prediction
            }

            results.append(result)

            print(f"Target Chunk : #{target_chunk}")
            print(f"Correct      : {correct}")
            print(f"Time         : {result['processing_time']:.2f}s")
            print("\nModel Response:")
            print(prediction)
            print()

        return results


# class LostInMiddleExperiment:

#     def __init__(self, system_prompt, query, answer):

#         self.system_prompt = system_prompt
#         self.query = query
#         self.answer = answer

#         self.generator = DummyDataGenerator(total_chunks=100)
#         self.inserter = AnswerInserter(answer)
#         self.context_assembler = ContextAssembler()
#         self.llm = OllamaLLM()

#         # Test the entire context window
#         # self.positions = [0.0, 0.1, 0.2, 0.3,
#         #                   0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#         self.positions = [0.0, 0.25, 0.5, 0.75, 1.0]

#     def evaluate(self, response):

#         return self.answer.lower() in response.lower()

#     def run(self):

#         print("\nRunning Lost-in-the-Middle Experiment...\n")

#         # Generate documents once
#         chunks = self.generator.generate(None)

#         results = []

#         for position in self.positions:

#             print("=" * 70)
#             print(f"Testing Position : {position:.1f}")
#             print("=" * 70)

#             # Insert answer
#             context = self.inserter.insert(
#                 chunks,
#                 position
#             )

#             # Build prompt
#             messages = self.context_assembler.assemble(
#                 system_prompt=self.system_prompt,
#                 user_prompt=self.query,
#                 retrieved_docs=context["chunks"]
#             )

#             # Query model
#             response = self.llm.generate(messages)

#             prediction = response["response"]

#             correct = self.evaluate(prediction)

#             result = {
#                 "position": position,
#                 "chunk_number": context["chunk_number"],
#                 "correct": correct,
#                 "processing_time": response["processing_time"],
#                 "prediction": prediction
#             }

#             results.append(result)

#             print(f"Chunk      : {result['chunk_number']}")
#             print(f"Correct    : {correct}")
#             print(f"Time       : {result['processing_time']:.2f}s")

#             print("\nModel Response:")
#             print(prediction)
#             print()

#         return results
