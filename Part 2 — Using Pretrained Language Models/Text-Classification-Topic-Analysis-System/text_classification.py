from transformers import pipeline


def classify_reviews(classifier, reviews):
    print("\n\nTEXT CLASSIFICATION")

    for index, review in enumerate(reviews, start=1):
        result = classifier(review)

        label = result[0]["label"]
        score = result[0]["score"]

        print(f"\nReview {index}:")
        print(review)
        print(f"Prediction: {label}")
        print(f"Confidence: {score:.2f}")


def main():
    classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

    reviews = [
        "The laptop is fast, reliable, and the battery lasts all day.",
        "The application is very slow and crashes frequently.",
        "The new update improved the performance significantly.",
        "The interface is confusing and difficult to use.",
        "The computer works perfectly and starts very quickly.",
        "The software has several bugs and keeps freezing.",
        "The keyboard feels comfortable and the build quality is excellent.",
        "The latest version made the application much worse.",
        "The system is easy to use and responds quickly.",
        "I am disappointed with the performance of this device."
    ]

    classify_reviews(classifier, reviews)


if __name__ == "__main__":
    main()