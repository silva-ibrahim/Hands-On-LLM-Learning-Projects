from transformers import AutoTokenizer, AutoModel
import torch


def analyze_attention(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    attentions = outputs.attentions
    last_layer_attention = attentions[-1][0]
    average_attention = last_layer_attention.mean(dim=0)
    return tokens, average_attention


def display_attention(tokens, attention, title):
    print("\n")
    print(title)
    print("-" * 30 + "\n")
    
    header = f"{'Token':<12}" + "".join([f"{t:>10}" for t in tokens])
    print(header)
    print("-" * len(header))
    
    for i, t1 in enumerate(tokens):
        row = f"{t1:<12}"
        for j, t2 in enumerate(tokens):
            score = attention[i][j].item()
            row += f"{score:>10.2f}"
        print(row)

def main():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModel.from_pretrained("distilbert-base-uncased", attn_implementation="eager")

    experiments = [
        ("Experiment 1 - Simple Sentence", "The cat sits on the mat."),
        ("Experiment 2 - Context", "The student studies computer engineering."),
        ("Experiment 3 - Repeated Words", "The model processes the text because the model understands context.")
    ]

    print("ATTENTION EXPERIMENT")
    print("Model: distilbert-base-uncased")

    for title, text in experiments:
        tokens, attention = analyze_attention(tokenizer, model, text)
        display_attention(tokens, attention, title)


if __name__ == "__main__":
    main()