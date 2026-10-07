from transformers import AutoTokenizer


def analyze_text(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    return tokens, token_ids


def run_experiment(tokenizer, title, original, modified):
    original_tokens, original_ids = analyze_text(tokenizer, original)
    modified_tokens, modified_ids = analyze_text(tokenizer, modified)

    print("\n")
    print(title)
    print("-" * 70)

    print("\nOriginal text:")
    print(original)
    print("Tokens:", original_tokens)
    print("Token IDs:", original_ids)
    print("Token count:", len(original_tokens))

    print("\nModified text:")
    print(modified)
    print("Tokens:", modified_tokens)
    print("Token IDs:", modified_ids)
    print("Token count:", len(modified_tokens))

    difference = len(modified_tokens) - len(original_tokens)

    print("\nToken count difference:", difference)


def main():
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

    experiments = [
        ("Experiment 1 - Capitalization", "machine learning is useful.", "Machine Learning Is Useful."),
        ("Experiment 2 - Punctuation", "Transformers process natural language", "Transformers process natural language!"),
        ("Experiment 3 - Technical terminology", "The model uses embeddings.", "The model uses contextualized embeddings."),
        ("Experiment 4 - Word length", "The system is efficient.", "The system is computationally efficient."),
        ("Experiment 5 - Numbers", "The model has 100 parameters.", "The model has 100000 parameters.")
    ]

    print("TOKENIZATION EXPERIMENT")
    print("Tokenizer: distilgpt2")

    for title, original, modified in experiments:
        run_experiment(tokenizer, title, original, modified)


if __name__ == "__main__":
    main()