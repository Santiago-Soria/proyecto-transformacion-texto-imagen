"""
build_share_umap.py
───────────────────
Genera data/shared/umap_params.pkl — fuente única de parámetros
normalizados [0,1] para las 3 técnicas de generación de imágenes.

Ubicación esperada: src/umap/build_share_umap.py

Reglas respetadas:
  • UMAP fit solo sobre X_train (sin leakage)
  • MinMaxScaler fit solo sobre X_train
  • np.clip en los 3 splits para garantizar rango [0, 1]
  • Contrato: 5 componentes, cosine, n_neighbors=15,
              min_dist=0.1, random_state=42
"""

import os
import joblib
import numpy as np
import umap
from sklearn.preprocessing import MinMaxScaler

# ── Rutas ────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RUTA_PKL  = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "models", "best_model",
                 "Exp_4.1_BETO_HPO_embeddings.pkl")
)
RUTA_UMAP = os.path.abspath(
    os.path.join(BASE_DIR, "..", "..", "data", "shared",
                 "umap_params.pkl")
)

# ── 1. Cargar embeddings del PKL ganador (Exp 4.1) ──────────
pkg = joblib.load(RUTA_PKL)

# Validación defensiva del PKL
for key in ('X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'):
    assert key in pkg, f"¡Falta '{key}' en el PKL! Regenerar con Notebook 3."

X_train = pkg['X_train']   # (908, 768)
X_val   = pkg['X_val']     # (114, 768)
X_test  = pkg['X_test']    # (114, 768)
y_train = pkg['y_train']
y_val   = pkg['y_val']
y_test  = pkg['y_test']

print(f"PKL cargado | X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

# ── 2. Entrenar UMAP solo sobre train ───────────────────────
reducer = umap.UMAP(
    n_components  = 5,
    n_neighbors   = 15,
    min_dist      = 0.1,
    metric        = 'cosine',
    random_state  = 42,
    n_jobs        = 1,
    low_memory    = False,
)
reducer.fit(X_train)
print("✓ UMAP ajustado sobre X_train")

# ── 3. Transformar los 3 splits ─────────────────────────────
params_train = reducer.transform(X_train)   # (908, 5)
params_val   = reducer.transform(X_val)     # (114, 5)
params_test  = reducer.transform(X_test)    # (114, 5)

# ── 4. Normalizar a [0, 1] — scaler fit solo en train ───────
scaler = MinMaxScaler()
scaler.fit(params_train)

# np.clip en los 3 splits:
#   - train: corrige ruido de punto flotante (ej. 1.0000000000000002)
#   - val/test: acota valores fuera de [0,1] por distribución distinta
# En ningún caso se modifica el fit del scaler → sin leakage.
params_train_norm = np.clip(scaler.transform(params_train), 0.0, 1.0)
params_val_norm   = np.clip(scaler.transform(params_val),   0.0, 1.0)
params_test_norm  = np.clip(scaler.transform(params_test),  0.0, 1.0)

# ── 5. Validación de rango ───────────────────────────────────
for name, arr in [('train', params_train_norm),
                  ('val',   params_val_norm),
                  ('test',  params_test_norm)]:
    assert arr.min() >= 0.0 and arr.max() <= 1.0, \
        f"¡{name} fuera de rango! [{arr.min():.16f}, {arr.max():.16f}]"
    print(f"  {name}: [{arr.min():.3f} – {arr.max():.3f}]  shape={arr.shape}")

# ── 6. Serializar ───────────────────────────────────────────
os.makedirs(os.path.dirname(RUTA_UMAP), exist_ok=True)

shared_pkg = {
    'reducer':      reducer,
    'scaler':       scaler,
    'train':        {'params': params_train_norm, 'labels': y_train},
    'val':          {'params': params_val_norm,   'labels': y_val},
    'test':         {'params': params_test_norm,  'labels': y_test},
    'n_components': 5,
    'metric':       'cosine',
    'random_state': 42,
}

joblib.dump(shared_pkg, RUTA_UMAP)
print(f"\n✓ umap_params.pkl guardado en: {RUTA_UMAP}")
print(f"  Tamaño: {os.path.getsize(RUTA_UMAP) / 1024:.1f} KB")