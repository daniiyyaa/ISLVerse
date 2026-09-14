import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

model_name = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

train_df = pd.read_csv("data/processed/train.csv")
valid_df = pd.read_csv("data/processed/valid.csv")

train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

prefix = "translate English to ISL gloss: "

def preprocess_function(examples):
    inputs = [
        prefix + sentence
        for sentence in examples["english"]
    ]

    targets = examples["gloss"]

    model_inputs = tokenizer(
        inputs,
        max_length=128,
        truncation=True
    )

    labels = tokenizer(
        text_target=targets,
        max_length=128,
        truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names
)

valid_dataset = valid_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=valid_dataset.column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model
)

training_args = Seq2SeqTrainingArguments(
    output_dir="models/mt5-isl-gloss",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=50,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    generation_max_length=64,
    logging_steps=10,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    processing_class=tokenizer,
    data_collator=data_collator
)

trainer.train()

trainer.save_model("models/mt5-isl-gloss")
tokenizer.save_pretrained("models/mt5-isl-gloss")

print("mT5 training completed successfully.")