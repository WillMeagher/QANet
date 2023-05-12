import json

import evaluate
import numpy as np
from transformers import DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline, AutoTokenizer
from datasets import Dataset, DatasetDict

data_file = "data/classification.small.buzztrain.json"

with open(data_file, "r") as infile:
    data_json = json.load(infile)

data = DatasetDict({
    "train": Dataset.from_dict(data_json["train"]),
    "test": Dataset.from_dict(data_json["test"])
})

file_path = "buzzer_model"

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.save_pretrained(file_path)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_data = data.map(preprocess_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

id2label = {0: "WRONG", 1: "RIGHT"}
label2id = {"WRONG": 0, "RIGHT": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)

training_args = TrainingArguments(
    output_dir="buzzer_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()