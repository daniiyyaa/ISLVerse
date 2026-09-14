import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "models/mt5-isl-gloss"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.eval()

while True:
    sentence = input("\nEnter an English sentence: ")

    if sentence.lower() == "exit":
        break

    input_text = "translate English to ISL gloss: " + sentence

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )

    gloss = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    print("Predicted ISL Gloss:", gloss.upper())