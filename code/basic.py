import argparse
import json
import pandas as pd
import os
import numpy as np
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model-name", default="cjvt/t5-sl-small", type=str, help="Model name"
)

parser.add_argument("--batch-size", default=4, type=int, help="Batch size")
parser.add_argument("--num-epochs", default=4, type=int, help="Number of epochs")
parser.add_argument("--learning-rate", default=1e-4, type=float, help="Learning rate")
parser.add_argument(
    "--output-dir",
    type=str,
    default="/d/hpc/projects/FRI/bb6846/",
    help="Directory to save the trained models and results",
)

parser.add_argument(
    "--data-dir",
    type=str,
    default="/d/hpc/home/bb6846/data/paragraphs/",
    help="Directory with data files",
)
parser.add_argument(
    "--train-file", type=str, default="train.jsonl", help="Training data filename"
)
parser.add_argument(
    "--val-file", type=str, default="validation.jsonl", help="Validation data filename"
)
parser.add_argument(
    "--token-len", type=int, default=512, help="size of the tokenized input"
)
args = parser.parse_args()

model_name = args.model_name


class DataEntry:
    def __init__(
        self,
        doc_id,
        src_grade,
        tgt_grade,
        src_sent_en,
        tgt_sent_en,
        pair_id,
        src_sent_sl,
        tgt_sent_sl,
    ):
        self.doc_id = doc_id
        self.src_grade = src_grade
        self.tgt_grade = tgt_grade
        self.src_sent_en = src_sent_en
        self.tgt_sent_en = tgt_sent_en
        self.pair_id = pair_id
        self.src_sent_sl = src_sent_sl
        self.tgt_sent_sl = tgt_sent_sl

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as file:
            data = [cls(**json.loads(line)) for line in file]
        return data

    def to_dict(self):
        return self.__dict__


# load everything
bleu = evaluate.load("bleu")
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Initialize the Sentence Transformer model.
labse_model = SentenceTransformer("sentence-transformers/LaBSE")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    try:
        bleu_result = bleu.compute(
            predictions=decoded_preds, references=[[t] for t in decoded_labels]
        )
        bleu_score = bleu_result["bleu"]
    except:
        bleu_score = 0
    try:
        embeddings1 = labse_model.encode(decoded_preds)
        embeddings2 = labse_model.encode(decoded_labels)

        avg_similarity = 0
        for i in range(len(embeddings1)):
            similarity = cosine_similarity([embeddings1[i]], [embeddings2[i]])[0][0]
            avg_similarity += similarity

        avg_similarity /= len(embeddings1)

    except:
        avg_similarity = 0

    return {"bleu": bleu_score, "LaBSE": avg_similarity}


def load_data():
    train_path = args.data_dir + args.train_file
    val_path = args.data_dir + args.val_file
    # Load data entries from JSON file
    data_entries_train = [
        x.to_dict() for x in DataEntry.from_json(json_file=train_path)
    ]
    data_entries_train = [
        {
            "id": f"{x['doc_id']}{x['src_grade']}{x['tgt_grade']}",
            "text": x["src_sent_sl"],
            "simplification": x["tgt_sent_sl"],
        }
        for x in data_entries_train
    ]

    data_entries_val = [x.to_dict() for x in DataEntry.from_json(json_file=val_path)]
    data_entries_val = [
        {
            "id": f"{x['doc_id']}{x['src_grade']}{x['tgt_grade']}",
            "text": x["src_sent_sl"],
            "simplification": x["tgt_sent_sl"],
        }
        for x in data_entries_val
    ]

    # Convert list of dictionaries to DataFrame
    df_train = pd.DataFrame(data_entries_train)
    df_val = pd.DataFrame(data_entries_val)

    # Convert DataFrame to dictionary of lists (column-wise data)
    data_dict_train = df_train.to_dict("list")
    data_dict_val = df_val.to_dict("list")

    # Convert dictionary to Hugging Face Dataset
    dataset_train = Dataset.from_dict(data_dict_train)
    dataset_val = Dataset.from_dict(data_dict_val)

    return dataset_train, dataset_val


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["text"], max_length=args.token_len, truncation=True
    )
    labels = tokenizer(
        examples["simplification"],
        max_length=args.token_len,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_data(train, val):
    train_dataset = train.map(preprocess_function, batched=True)
    val_dataset = val.map(preprocess_function, batched=True)

    return train_dataset, val_dataset


# Load data entries from JSON file
train_dataset, val_dataset = load_data()

train_dataset, val_dataset = preprocess_data(train_dataset, val_dataset)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"{args.output_dir}/{model_name.split('/')[-1]}",
    evaluation_strategy="epoch",
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=args.num_epochs,
    gradient_accumulation_steps=4,
    predict_with_generate=True,
    optim="adafactor",
    load_best_model_at_end=True,
    save_strategy="epoch",
    metric_for_best_model="LaBSE",
    # fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# trainer.train(latest_checkpoint if latest_checkpoint else None)
trainer.train()

trainer.save_model(f"{args.output_dir}{model_name.split('/')[-1]}-TS-fine-tuned")
print(f"model saved in {args.output_dir}{model_name.split('/')[-1]}-TS-fine-tuned")
