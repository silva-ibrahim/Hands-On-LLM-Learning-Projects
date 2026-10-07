from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def create_embeddings(model, texts):
    return model.encode(texts, convert_to_tensor=True)

def semantic_search(model, texts, embeddings, query, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(query_embedding.unsqueeze(0), embeddings)
    top_results = torch.topk(similarities, k=top_k)
    results = []
    for index in top_results.indices:
        results.append(texts[index.item()])
    return results

def generate_answer(tokenizer, model, query, relevant_texts):
    context = "\n".join(relevant_texts)
    prompt = "Answer the question in one complete sentence using only the information below.\n\nInformation:\n" + context + "\n\nQuestion: " + query + "\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60, min_new_tokens=8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def run_experiment(embedding_model, tokenizer, generation_model, texts, query, experiment_number):
    embeddings = create_embeddings(embedding_model, texts)
    relevant_texts = semantic_search(embedding_model, texts, embeddings, query)
    answer = generate_answer(tokenizer, generation_model, query, relevant_texts)

    print(f"\nRAG EXPERIMENT {experiment_number}")
    print(f"\nQuestion: {query}")
    print("\nRetrieved Information:")
    for index, text in enumerate(relevant_texts, start=1):
        print(f"{index}. {text}")
    print(f"\nGenerated Answer: {answer}")

def main():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    generation_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    texts = [
        "Python is a high-level programming language used for software development, data analysis, and artificial intelligence.",
        "Machine learning allows computers to learn patterns from data and make predictions without being explicitly programmed for every task.",
        "Transformer models use attention mechanisms to process relationships between tokens in a sequence.",
        "Computer networks allow devices to communicate and exchange data with each other.",
        "Natural language processing enables computers to process and analyze human language.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns from data.",
        "Tokenization is the process of dividing text into smaller units called tokens.",
        "Embeddings represent text as numerical vectors that can be compared using similarity measures."
    ]

    experiments = [
        "What is a Transformer model?",
        "What is machine learning?",
        "How does tokenization work?"
    ]

    print("RAG-BASED SEMANTIC SEARCH ASSISTANT")
    print("Embedding Model: all-MiniLM-L6-v2")
    print("Generation Model: google/flan-t5-small")

    for index, query in enumerate(experiments, start=1):
        run_experiment(embedding_model, tokenizer, generation_model, texts, query, index)

if __name__ == "__main__":
    main()