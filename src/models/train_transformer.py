import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset

class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", 
                           max_length=128, return_tensors="pt")
        return {
            'input_ids': enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def run_finetuning(train_texts, train_labels, val_texts, val_labels, model_name="PlanTL-GOB-ES/roberta-base-bne"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    train_ds = DepressionDataset(train_texts, train_labels, tokenizer)
    val_ds = DepressionDataset(val_texts, val_labels, tokenizer)
    
    args = TrainingArguments(
        safe_model_name = model_name.replace("/", "_"),
        output_dir=f"models/checkpoints/{safe_model_name}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=4,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
    )
    
    trainer.train()
    return trainer