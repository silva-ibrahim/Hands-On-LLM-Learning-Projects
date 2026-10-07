from collections import Counter

questions = ["What is a CPU?", "What is RAM?", "What is cache memory?"]

expected_answers = [
    "A CPU is a processor that executes instructions.",
    "RAM is temporary memory used by running programs.",
    "Cache memory is a small and fast memory near the CPU."
]

model_answers = [
    "A CPU is the main processor that executes computer instructions.",
    "RAM temporarily stores data used by running applications.",
    "Cache memory is a small and fast memory close to the CPU."
]


def rouge_1(expected, answer):
    expected_words = expected.lower().split()
    answer_words = answer.lower().split()
    expected_count = Counter(expected_words)
    answer_count = Counter(answer_words)
    common_words = sum(min(expected_count[word], answer_count[word]) for word in expected_count)

    precision = common_words / len(answer_words)
    recall = common_words / len(expected_words)

    if precision + recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)


print("LLM Response Evaluation Experiment")

for question, expected, answer in zip(questions, expected_answers, model_answers):
    score = rouge_1(expected, answer)
    print(f"\nQuestion: {question}\nExpected: {expected}\nAnswer: {answer}\nROUGE-1 F1: {score:.2f}")

print("\nEvaluation completed successfully.")