from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from src.preprocessing_utils import preprocess_text 

def run_exp_1_1(X_train, y_train, X_test, y_test):
    """
    Exp 1.1: Limpieza P1 + TF-IDF + Regresión Logística
    """
    # 1. Definir el Pipeline
    # Usamos n-gramas (1,2) para capturar algo de contexto (ej. "no quiero")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', LogisticRegression(class_weight='balanced', solver='liblinear'))
    ])
    
    # 2. Preprocesamiento (Aplicar P1)
    # Nota: Es mejor aplicar la limpieza ANTES de entrar al pipeline si es costosa
    X_train_clean = [preprocess_text(text, method='P1') for text in X_train]
    X_test_clean = [preprocess_text(text, method='P1') for text in X_test]
    
    # 3. Entrenar
    print("Entrenando Exp 1.1...")
    pipeline.fit(X_train_clean, y_train)
    
    # 4. Evaluar
    predictions = pipeline.predict(X_test_clean)
    print(classification_report(y_test, predictions))
    
    return pipeline