from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.sentence_transformer import losses
from datasets import Dataset
import torch


def similarity(model, text1, text2):
    embedding1 = model.encode(text1, convert_to_tensor=True)
    embedding2 = model.encode(text2, convert_to_tensor=True)
    return torch.nn.functional.cosine_similarity(embedding1, embedding2, dim=0).item()


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    train_dataset = Dataset.from_dict({
        "anchor": [
            "How does a CPU execute instructions?",
            "What is RAM used for?",
            "What is a computer network?",
            "What does a compiler do?",
            "What is cache memory?"
        ],
        "positive": [
            "A processor executes instructions by fetching and processing them.",
            "RAM temporarily stores data used by running programs.",
            "A computer network allows devices to communicate with each other.",
            "A compiler translates source code into machine code.",
            "Cache is a small and fast memory located close to the CPU."
        ]
    })

    test_pairs = [
        ("How does a processor work?", "A CPU executes instructions."),
        ("Where is temporary data stored?", "RAM stores data for running programs."),
        ("How do computers communicate?", "Networks allow computers to communicate.")
    ]

    print("Domain-Specific Embedding Fine-Tuning")

    print("\nBefore fine-tuning")

    for name, (text1, text2) in zip(["CPU", "RAM", "Network"], test_pairs):
        print(f"{name}: {similarity(model, text1, text2):.2f}")

    loss = losses.MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir="./fine-tuned-model",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        learning_rate=2e-5
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss
    )

    print("\nFine-tuning model...")
    trainer.train()

    print("\nAfter fine-tuning")

    for name, (text1, text2) in zip(["CPU", "RAM", "Network"], test_pairs):
        print(f"{name}: {similarity(model, text1, text2):.2f}")

    model.save("./fine-tuned-model")

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()