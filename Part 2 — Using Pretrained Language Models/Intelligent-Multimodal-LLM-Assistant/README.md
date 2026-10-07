# Intelligent Multimodal LLM Assistant

A Python-based multimodal application developed to explore image understanding and visual question answering using a pretrained vision-language model.

This project was created to reinforce Python programming and multimodal language model concepts through practical development.

The application processes an input image and answers questions about its content using the `Salesforce/blip-vqa-base` model.

## Features

- Process an input image
- Ask questions about the image
- Generate answers based on visual information
- Experiment with different image-related questions
- Use a pretrained vision-language model
- Automatically use GPU when available

## Technologies Used

- Python
- Hugging Face Transformers
- PyTorch
- BLIP VQA (`Salesforce/blip-vqa-base`)
- Visual Studio Code
- Google Colab

## Project Structure

```text
Intelligent-Multimodal-LLM-Assistant/
├── README.md
├── requirements.txt
├── multimodal_assistant.py
└── screenshots/
    ├── multimodal-assistant-output.png
    └── input-image.png
```

## How to Run

### 1. Install the Required Dependencies

```bash
!pip install -r requirements.txt
```

### 2. Upload the Input Image

Make sure to upload an image named `input-image.png.jpg` to your Google Colab environment before running the script.

The image should be available in the `/content/` directory.

### 3. Run the Multimodal Assistant

```bash
!python multimodal_assistant.py
```

The project was developed using Visual Studio Code and executed using Google Colab.

## Screenshots

### Multimodal Assistant Output

![Multimodal Assistant Output](screenshots/multimodal-assistant-output.png)

### Input Image

![Input Image](screenshots/input-image.png)

## Related Learning

This project was developed as part of my learning journey with *Hands-On Large Language Models* by Jay Alammar and Maarten Grootendorst.

The project was developed alongside my study of the multimodal and visual language model concepts covered in the book, allowing me to apply these concepts through a practical experiment.

The experiment focuses on processing an image and using a pretrained vision-language model to answer questions about the visual content.

## Author

**Silva Ibrahim**  
Computer Engineering Student

## License

This project is licensed under the MIT License.