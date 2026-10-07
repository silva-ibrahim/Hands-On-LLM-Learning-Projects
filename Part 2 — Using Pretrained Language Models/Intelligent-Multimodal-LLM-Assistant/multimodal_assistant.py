from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

def load_image():
    image = Image.open("/content/input-image.png.jpg").convert("RGB")
    return image

def ask_questions(processor, model, image, questions, device):
    print("\nMULTIMODAL LLM ASSISTANT")

    for index, question in enumerate(questions, start=1):
        inputs = processor(images=image, text=question, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=30)

        answer = processor.decode(output[0], skip_special_tokens=True)

        print(f"\nQuestion {index}: {question}")
        print(f"Answer: {answer}")

def main():
    model_id = "Salesforce/blip-vqa-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(model_id)
    model.to(device)

    image = load_image()

    questions = [
        "What is in the image?",
        "What color is the main object?",
        "Where is the object?"
    ]

    print("INTELLIGENT MULTIMODAL LLM ASSISTANT")
    print(f"Model: {model_id}")

    ask_questions(processor, model, image, questions, device)

if __name__ == "__main__":
    main()