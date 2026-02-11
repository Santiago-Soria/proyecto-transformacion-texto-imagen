import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from tqdm import tqdm

class BertFeatureExtractor:
    def __init__(self, model_name="dccuchile/bert-base-spanish-wwm-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Modo evaluación (congela dropout)

    def get_embeddings(self, texts, batch_size=32):
        all_embeddings = []
        
        # Procesar por lotes para no saturar la VRAM
        for i in tqdm(range(0, len(texts), batch_size), desc="Extrayendo Embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenización
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Usamos el token [CLS] (primero de la secuencia) como representación de la frase
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)
                
        return np.vstack(all_embeddings)

# En el notebook:
# extractor = BertFeatureExtractor()
# X_train_emb = extractor.get_embeddings(X_train_clean)
# clf = LogisticRegression()
# clf.fit(X_train_emb, y_train)