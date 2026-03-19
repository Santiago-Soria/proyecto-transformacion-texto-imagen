import os
import json
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def train_logistic(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    experiment_name="exp_generic",
    models_dir=None
):
    
    model_path = os.path.join(models_dir, f"{experiment_name}.pkl")
    results_dir = os.path.join(models_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nEntrenando Regresión Logística para: {experiment_name}")

    # ==============================
    # Entrenamiento (sobre train)
    # ==============================
    clf = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        random_state=42,
        max_iter=1000
    )

    clf.fit(X_train, y_train)

    # ==============================
    # Validación (sobre validation)
    # ==============================
    preds_val = clf.predict(X_val)
    f1_val = f1_score(y_val, preds_val, average="macro")

    # ==============================
    # Evaluación (sobre test)
    # ==============================
    preds_test = clf.predict(X_test)
    precision, recall, f1_test, _ = precision_recall_fscore_support(
    y_test, preds_test, average='macro')
    accuracy = accuracy_score(y_test, preds_test)

    print(f"\n{'='*50}")
    print(f"  {experiment_name}")
    print(f"{'='*50}")
    print(f"  Val  F1-Macro:  {f1_val:.4f}")
    print(f"  Test Accuracy:  {accuracy:.4f}")
    print(f"  Test Precision: {precision:.4f}")
    print(f"  Test Recall:    {recall:.4f}")
    print(f"  Test F1-Macro:  {f1_test:.4f}  ← métrica principal")
    print(f"\n{classification_report(y_test, preds_test, target_names=['No Dep','Dep'], zero_division=0)}")

    # Matriz de Confusión
    cm = confusion_matrix(y_test, preds_test)
    print("\nConfusion Matrix:")
    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Dep", "Dep"],
        yticklabels=["No Dep", "Dep"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {experiment_name}")

    plt.show()    
    plt.savefig(os.path.join(results_dir, f"{experiment_name}_confusion_matrix.png"))

    # Matríz de Confusión Normalizada
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["No Dep", "Dep"],
        yticklabels=["No Dep", "Dep"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Normalized Confusion Matrix - {experiment_name}")

    plt.show()    
    plt.savefig(os.path.join(results_dir, f"{experiment_name}_confusion_matrix_norm.png"))


    # ==============================
    # Métricas estructuradas
    # ==============================
    metrics = {
        'experiment':       experiment_name,
        'f1_macro_val':     round(f1_val, 4),
        'f1_macro_test':    round(f1_test, 4),
        'accuracy_test':    round(accuracy, 4),
        'precision_macro':  round(precision, 4),
        'recall_macro':     round(recall, 4),
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
    joblib.dump(clf, model_path)

    print(f"\nModelo guardado en: {model_path}")

    # ==============================
    # Guardar métricas (JSON)
    # ==============================
    results_path = os.path.join(results_dir, f"{experiment_name}_metrics.json")

    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Métricas guardadas en: {results_path}")

    # ==============================
    # Retorno estructurado
    # ==============================
    return clf, preds_test, metrics