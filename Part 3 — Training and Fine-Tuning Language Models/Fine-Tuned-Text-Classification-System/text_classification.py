from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np


texts = [
    "The CPU executes program instructions.",
    "The processor performs calculations.",
    "The CPU processes data and instructions.",
    "The processor controls program execution.",
    "The CPU performs arithmetic operations.",
    "RAM stores data temporarily.",
    "RAM is used by running programs.",
    "Computer memory stores active data.",
    "RAM provides temporary storage.",
    "Memory holds data for active applications.",
    "A network connects different devices.",
    "Computer networks allow data communication.",
    "A network connects computers together.",
    "Networks allow devices to exchange data.",
    "Computers communicate through networks."
]

labels = [
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2
]

dataset = Dataset.from_dict({"text": texts, "label": labels})

model_name = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize(data):
    return tokenizer(data["text"], padding="max_length", truncation=True, max_length=32)


dataset = dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=5,
    per_device_train_batch_size=5,
    learning_rate=2e-5,
    report_to="none"
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)

print("Fine-Tuned Text Classification System")
print("\nFine-tuning model...")
trainer.train()

test_texts = [
    "The processor handles program instructions.",
    "Temporary memory is used by active programs.",
    "Devices exchange information through a network."
]

inputs = tokenizer(
    test_texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

outputs = model(**inputs)
predictions = np.argmax(outputs.logits.detach().numpy(), axis=1)

categories = ["CPU", "RAM", "Network"]

print("\nPredictions")

for text, prediction in zip(test_texts, predictions):
    print(f"Text: {text}")
    print(f"Category: {categories[prediction]}")
    print()

model.save_pretrained("./fine-tuned-model")
tokenizer.save_pretrained("./fine-tuned-model")

print("Model saved successfully.")