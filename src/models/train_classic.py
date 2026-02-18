from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
import joblib
import os

def train_logistic(X_train, y_train, X_test, y_test, experiment_name="exp_generic", models_dir=None):
    print(f"Entrenando Regresión Logística para {experiment_name}...")
    
    # Entrenamiento
    clf = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluación
    preds = clf.predict(X_test)
    score = f1_score(y_test, preds, average='macro')
    
    # Reporte
    print(f"--- Resultados {experiment_name} ---")
    print(classification_report(y_test, preds))
    print(f"\n---> Resultado (Macro-F1) - {experiment_name}: {score:.4f}")

    # Guardar modelo - Compatible con local y Colab
    if models_dir is None:
        # Intentar detectar el entorno
        try:
            # Si estamos en un archivo .py (local)
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "..", "..", "models")
        except NameError:
            # Si estamos en notebook (Colab o Jupyter)
            # Buscar la carpeta models desde el directorio actual
            if os.path.exists('/content/proyecto-transformacion-texto-imagen'):
                # Estamos en Colab
                models_dir = '/content/proyecto-transformacion-texto-imagen/models'
            else:
                # Estamos en otro notebook, usar directorio actual
                models_dir = './models'
    
    # Crear directorio si no existe
    os.makedirs(models_dir, exist_ok=True)
    
    # Guardar modelo
    model_path = os.path.join(models_dir, f"{experiment_name}.pkl")
    joblib.dump(clf, model_path)
    print(f"✓ Modelo guardado en: {model_path}")
    
    return clf, preds, score
