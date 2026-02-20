import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_tfidf(self, texts, max_features=5000, ngram_range=(1, 2)):
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X = vectorizer.fit_transform(texts)
        return X, vectorizer

    def get_transformer_embeddings(self, texts, model_name="PlanTL-GOB-ES/roberta-base-bne", batch_size=32):
        print(f"--> Extrayendo embeddings con {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        model.eval()

        all_embeddings = []
        
        # Procesamiento por lotes para no saturar memoria
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, 
                             max_length=128, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # Usamos el token [CLS] (o su equivalente <s> en RoBERTa)
                cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_emb)
        
        return np.vstack(all_embeddings)