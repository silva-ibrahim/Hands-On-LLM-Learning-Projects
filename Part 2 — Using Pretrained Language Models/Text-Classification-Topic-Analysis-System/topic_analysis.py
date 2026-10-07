from sentence_transformers import SentenceTransformer
import hdbscan


def analyze_topics(model, texts):
    embeddings = model.encode(texts)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric="euclidean")

    labels = clusterer.fit_predict(embeddings)

    return labels


def display_topics(texts, labels):
    print("\n\nTOPIC ANALYSIS")

    topics = {}

    for index, label in enumerate(labels):
        if label not in topics:
            topics[label] = []

        topics[label].append(texts[index])

    for label, topic_texts in topics.items():
        if label == -1:
            print("\nOutliers:")
        else:
            print(f"\nTopic {label + 1}:")

        for text in topic_texts:
            print("-", text)


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [
        "The laptop battery lasts all day.",
        "The computer has a powerful processor.",
        "The laptop performance is very fast.",
        "The battery charges quickly.",
        "The new software update improves security.",
        "The application has several security features.",
        "The system protects user data.",
        "The software includes better privacy controls.",
        "The keyboard is comfortable to use.",
        "The laptop screen has excellent image quality."
    ]

    labels = analyze_topics(model, texts)
    display_topics(texts, labels)


if __name__ == "__main__":
    main()