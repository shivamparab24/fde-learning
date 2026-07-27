from DO1.test.lost_in_middle import LostInMiddleExperiment


def main():

    system_prompt = (
        "You are a helpful assistant. "
        "Answer ONLY using the provided documents. "
        "If the answer is not present, say 'I don't know.'"
    )

    query = "How many sick leave days are provided annually?"

    answer = "."

    experiment = LostInMiddleExperiment(
        system_prompt=system_prompt,
        query=query,
        answer=answer
    )

    results = experiment.run()

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    correct_count = 0

    for result in results:

        if result["correct"]:
            correct_count += 1

        print(
            f"Position: {result['position']:.1f} | "
            f"Chunk: {result['chunk_number']:3d} | "
            f"Correct: {result['correct']} | "
            f"Time: {result['processing_time']:.2f}s"
        )

    print("\n" + "=" * 80)
    print(
        f"Accuracy: {correct_count}/{len(results)} "
        f"({100 * correct_count / len(results):.1f}%)"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
