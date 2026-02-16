from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import joblib

def train_logistic(X_train, y_train, X_test, y_test, experiment_name="exp_generic"):
    print(f"Entrenando Regresión Logística para {experiment_name}...")
    
    # Class_weight='balanced' es CRÍTICO por tu desbalance 60/40
    clf = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    clf.fit(X_train, y_train)
    
    preds = clf.predict(X_test)
    
    # Reporte
    print(f"--- Resultados {experiment_name} ---")
    print(classification_report(y_test, preds))
    
    # Guardar modelo
    joblib.dump(clf, f"models/{experiment_name}.pkl")
    return clf, preds