import json
import random


class DummyDataGenerator:

    def __init__(self, total_chunks=100):
        self.total_chunks = total_chunks

        self.answer = "Employees are entitled to 12 sick leave days annually."

        self.filler = [
            # "Employees must submit leave requests through the HR portal.",
            # "Annual leave must be approved by the reporting manager.",
            # "The company offers work-from-home options for eligible employees.",
            # "Medical certificates are required for extended sick leave.",
            # "Unused casual leave expires at the end of the calendar year.",
            # "Payroll is processed on the final working day of each month.",
            # "Employees can update personal information using the HR system.",
            # "Training sessions are available every quarter.",
            # "Performance reviews are conducted twice a year.",
            # "The HR department maintains employee attendance records.",
            # "Company holidays are announced at the beginning of each year.",
            # "Employees should notify their manager before taking emergency leave.",
            # "Overtime requires prior approval from department heads.",
            # "New employees complete an onboarding program during their first week.",
            # "Travel reimbursement claims should be submitted within 30 days.",
            # "Managers are responsible for approving expense reports.",
            # "Employees can access payroll statements online.",
            # "The company encourages continuous learning and skill development.",
            # "Health insurance benefits begin after the probation period.",
            # "Flexible working hours are available for eligible teams."
            "The standard operating procedure dictates that all inventory forms must be signed in triplicate by a regional manager. ",
            "Ensure that the air conditioning units are set to precisely 72 degrees Fahrenheit during weekends to conserve energy. ",
            "The logistics team reports that shipping delays may occur if customs paperwork is missing an official government seal. ",
            "Quarterly fiscal statements must be submitted no later than the final working day of the month to prevent compliance penalties. ",
            "All employee badges must remain clearly visible above the waist while navigating the secure sectors of the main campus. ",
            "IT support recommends clearing your browser cache and resetting your local network hardware if connection lag persists. ",
            "The annual corporate retreat has been tentatively rescheduled for the first week of October pending budget finalization.",
 
        ]

    # def generate(self):

    #     chunks = []

    #     for i in range(1, self.total_chunks + 1):

    #         # Each chunk gets 5 random HR sentences
    #         paragraph = " ".join(random.sample(self.sentences, 3))

    #         chunks.append(
    #             {
    #                 "id": i,
    #                 "chunk": paragraph
    #             }
    #         )

    #     return chunks

    def generate(self, position):
        """Generates a single context with the needle at the given position ratio."""
        chunks = []

        if isinstance(position, (int, float)):
            target_slot = min(int(position * self.total_chunks), self.total_chunks - 1)
        else:
            target_slot = -1  # Filler only

        filler_paragraph = " ".join(self.filler)

        for i in range(self.total_chunks):
            if i == target_slot:
                paragraph = self.answer
                is_needle = True
            else:
                paragraph = f"[ID: {i}] {filler_paragraph}"
                is_needle = False

            chunks.append(
                {
                    "id": i + 1,
                    "chunk": paragraph,
                    "is_needle": is_needle
                }
            )

        return chunks
    
    def generate_all_contexts(self, positions):
        """Generates a separate context dictionary for each position in the list."""
        all_contexts = {}
        for pos in positions:
            all_contexts[pos] = self.generate(position=pos)
        return all_contexts


    def save(self, chunks, file_path="DO1/data/dummy_chunks.json"):

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=4)

        print(f"Saved {len(chunks)} chunks.")


# if __name__ == "__main__":
#     generator = DummyDataGenerator()
#     chunks = generator.generate()
#     generator.save(chunks)
