import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_path = "models/mt5-isl-gloss"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

test_df = pd.read_csv("data/processed/test.csv")

correct = 0
total = len(test_df)

for _, row in test_df.iterrows():
    sentence = str(row["english"])
    expected_gloss = str(row["gloss"]).strip().upper()

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
            max_length=64,
            num_beams=4
        )

    predicted_gloss = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    ).strip().upper()

    if predicted_gloss == expected_gloss:
        correct += 1

    print("\nEnglish:", sentence)
    print("Expected:", expected_gloss)
    print("Predicted:", predicted_gloss)

accuracy = (correct / total) * 100

print("\n----------------------------")
print("Total test sentences:", total)
print("Correct predictions:", correct)
print(f"Exact-match accuracy: {accuracy:.2f}%")
print("----------------------------")