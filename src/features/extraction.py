import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_tfidf(self, X_train, X_val, X_test, max_features=5000, ngram_range=(1, 2)):
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
        X_train_tfidf = vectorizer.fit_transform(X_train)  # aprende vocabulario de train
        X_val_tfidf   = vectorizer.transform(X_val)        # solo transforma, no aprende
        X_test_tfidf  = vectorizer.transform(X_test)       # solo transforma, no aprende
        return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer

    def get_frozen_embeddings(self, texts, model_name, batch_size=32):
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