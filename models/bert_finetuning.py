from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
from torch.utils.data import Dataset
import torch

class DepressionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_bert_finetuning(X_train, y_train, X_val, y_val):
    model_name = "dccuchile/bert-base-spanish-wwm-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Preparar Datasets
    train_dataset = DepressionDataset(X_train, y_train, tokenizer)
    val_dataset = DepressionDataset(X_val, y_val, tokenizer)
    
    # Cargar modelo pre-entrenado para clasificación (agrega la capa densa automáticamente)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    # Configuración de entrenamiento (Best Practices)
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,              # BERT aprende rápido, 3-5 épocas suelen bastar
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,                # Calentamiento para no romper los pesos pre-entrenados
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",     # Evaluar al final de cada época
        save_strategy="epoch",           # Guardar el mejor modelo
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    trainer.train()
    return trainer