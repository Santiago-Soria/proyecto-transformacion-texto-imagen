import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)

# ==============================
# Reproducibilidad
# ==============================
set_seed(42)


# ==============================
# Dataset
# ==============================
class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


# ==============================
# Métricas
# ==============================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


# ==============================
# Fine-Tuning Principal
# ==============================
def run_finetuning(train_texts, train_labels, val_texts, val_labels, model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    train_ds = DepressionDataset(train_texts, train_labels, tokenizer)
    val_ds = DepressionDataset(val_texts, val_labels, tokenizer)

    safe_model_name = model_name.replace("/", "_")
    output_dir = f"models/checkpoints/{safe_model_name}"

    training_args = TrainingArguments(
        output_dir=output_dir,

        # Estrategia de evaluación
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",

        # Optimización
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=6,
        weight_decay=0.01,

        # Selección del mejor modelo
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,

        # Control de checkpoints
        save_total_limit=1,

        # Reproducibilidad
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()

    # ==========================
    # Evaluación Final
    # ==========================
    final_metrics = trainer.evaluate()
    print("\nFinal Evaluation Metrics:")
    print(final_metrics)

    # Guardar métricas en JSON
    os.makedirs("results", exist_ok=True)
    with open(f"results/{safe_model_name}_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    # ==========================
    # Matriz de Confusión
    # ==========================
    predictions = trainer.predict(val_ds)
    preds = np.argmax(predictions.predictions, axis=1)
    cm = confusion_matrix(val_labels, preds)

    print("\nConfusion Matrix:")
    print(cm)

    return trainer