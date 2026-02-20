import os
import json
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


def train_logistic(
    X_train,
    y_train,
    X_test,
    y_test,
    experiment_name="exp_generic",
    models_dir=None
):
    print(f"\nEntrenando Regresión Logística para: {experiment_name}")

    # ==============================
    # Entrenamiento
    # ==============================
    clf = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
        max_iter=1000
    )

    clf.fit(X_train, y_train)

    # ==============================
    # Evaluación
    # ==============================
    preds = clf.predict(X_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        preds,
        average="macro"
    )

    accuracy = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    print("\nResultados:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Macro): {precision:.4f}")
    print(f"Recall (Macro): {recall:.4f}")
    print(f"F1 (Macro): {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\n📄 Classification Report:")
    print(classification_report(y_test, preds))

    # ==============================
    # Métricas estructuradas
    # ==============================
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1)
    }

    # ==============================
    # Manejo de rutas (Local / Colab)
    # ==============================
    if models_dir is None:
        try:
            # Si estamos en script local
            base_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(base_dir, "..", "..", "models")
        except NameError:
            # Notebook
            if os.path.exists('/content/proyecto-transformacion-texto-imagen'):
                models_dir = '/content/proyecto-transformacion-texto-imagen/models'
            else:
                models_dir = './models'

    os.makedirs(models_dir, exist_ok=True)

    # ==============================
    # Guardar modelo
    # ==============================
    model_path = os.path.join(models_dir, f"{experiment_name}.pkl")
    joblib.dump(clf, model_path)

    print(f"\nModelo guardado en: {model_path}")

    # ==============================
    # Guardar métricas (JSON)
    # ==============================
    results_dir = os.path.join(models_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    results_path = os.path.join(results_dir, f"{experiment_name}_metrics.json")

    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas guardadas en: {results_path}")

    # ==============================
    # Retorno estructurado
    # ==============================
    return clf, preds, metrics