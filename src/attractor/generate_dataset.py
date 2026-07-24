"""
generate_dataset.py
───────────────────
Genera el dataset completo de imágenes Fase 1 (puras) para
Atractores de Lorenz a partir del PKL compartido umap_params.pkl.

Ubicación: src/attractor/generate_dataset.py

Salidas:
  data/images/attractor_lorenz/
  ├── train/
  │   ├── train_0000.png   (908 imágenes)
  │   └── ...
  ├── val/
  │   ├── val_0000.png     (114 imágenes)
  │   └── ...
  └── test/
      ├── test_0000.png    (114 imágenes)
      └── ...

  data/shared/
  ├── metadata_attractor_train.csv
  ├── metadata_attractor_val.csv
  └── metadata_attractor_test.csv

Cada CSV de metadata contiene:
  filename, split, label, sigma, rho, beta, n_puntos, elevacion

Uso:
  cd proyecto-transformacion-texto-imagen
  python src/attractor/generate_dataset.py
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Rutas ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

RUTA_UMAP_PKL = os.path.join(PROJECT_ROOT, 'data', 'shared', 'umap_params.pkl')
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'data', 'images', 'attractor_lorenz')
METADATA_DIR  = os.path.join(PROJECT_ROOT, 'data', 'shared')

# ── Importar lorenz.py (mismo directorio) ────────────────────
sys.path.insert(0, BASE_DIR)
from lorenz import (
    map_umap_to_lorenz,
    solve_attractor,
    render_attractor_pure,
)


# ─────────────────────────────────────────────────────────────
# Generación por split
# ─────────────────────────────────────────────────────────────

def generate_split(params_norm, labels, split_name, output_base):
    """
    Genera imágenes Fase 1 y metadata para un split completo.

    Parameters
    ----------
    params_norm : ndarray, shape (N, 5)
        Componentes UMAP normalizados [0, 1].
    labels : ndarray, shape (N,)
        Etiquetas (0 = no_depresivo, 1 = depresivo).
    split_name : str
        Nombre del split ('train', 'val', 'test').
    output_base : str
        Directorio base para imágenes (se crea subcarpeta por split).

    Returns
    -------
    list of dict
        Registros de metadata para el CSV.
    """
    split_dir = os.path.join(output_base, split_name)
    os.makedirs(split_dir, exist_ok=True)

    n_samples = len(labels)
    metadata = []
    errores = []

    print(f"\n{'─' * 60}")
    print(f"  Generando split: {split_name} ({n_samples} muestras)")
    print(f"  Directorio: {split_dir}")
    print(f"{'─' * 60}")

    t_start = time.time()

    for i in range(n_samples):
        filename = f"{split_name}_{i:04d}.png"
        filepath = os.path.join(split_dir, filename)

        try:
            # 1. Mapear UMAP → parámetros de Lorenz
            umap_components = params_norm[i]
            params = map_umap_to_lorenz(umap_components)

            # 2. Resolver ODE
            trajectory = solve_attractor(
                sigma=params['sigma'],
                rho=params['rho'],
                beta=params['beta'],
                n_puntos=params['n_puntos'],
            )

            # 3. Renderizar imagen pura
            label = int(labels[i])
            fig = render_attractor_pure(
                trajectory=trajectory,
                elevacion=params['elevacion'],
                output_path=filepath,
            )
            plt.close(fig)  # liberar memoria

            # 4. Registrar metadata
            metadata.append({
                'filename':  filename,
                'split':     split_name,
                'label':     label,
                'sigma':     round(params['sigma'], 4),
                'rho':       round(params['rho'], 4),
                'beta':      round(params['beta'], 4),
                'n_puntos':  params['n_puntos'],
                'elevacion': round(params['elevacion'], 2),
                'umap_0':    round(float(umap_components[0]), 6),
                'umap_1':    round(float(umap_components[1]), 6),
                'umap_2':    round(float(umap_components[2]), 6),
                'umap_3':    round(float(umap_components[3]), 6),
                'umap_4':    round(float(umap_components[4]), 6),
            })

            # Progreso
            if (i + 1) % 50 == 0 or (i + 1) == n_samples:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (n_samples - i - 1) / rate if rate > 0 else 0
                print(f"    [{i+1:>4d}/{n_samples}] "
                      f"{elapsed:.0f}s transcurridos | "
                      f"{rate:.1f} img/s | "
                      f"ETA: {eta:.0f}s")

        except Exception as e:
            errores.append({'index': i, 'filename': filename, 'error': str(e)})
            print(f"    ⚠ Error en {filename}: {e}")

    elapsed_total = time.time() - t_start
    print(f"\n  ✓ {split_name} completado: {len(metadata)} imágenes "
          f"en {elapsed_total:.1f}s")

    if errores:
        print(f"  ⚠ {len(errores)} errores:")
        for err in errores:
            print(f"     - {err['filename']}: {err['error']}")

    return metadata


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Generador de Dataset — Atractores de Lorenz (Fase 1)")
    print("=" * 60)

    # ── 1. Cargar PKL compartido ─────────────────────────────
    assert os.path.exists(RUTA_UMAP_PKL), \
        f"No se encontró umap_params.pkl en: {RUTA_UMAP_PKL}"

    pkg = joblib.load(RUTA_UMAP_PKL)
    print(f"\n✓ umap_params.pkl cargado desde: {RUTA_UMAP_PKL}")

    # Validar estructura
    for split in ('train', 'val', 'test'):
        assert split in pkg, f"Falta split '{split}' en el PKL"
        assert 'params' in pkg[split], f"Falta 'params' en pkg['{split}']"
        assert 'labels' in pkg[split], f"Falta 'labels' en pkg['{split}']"

    # ── 2. Extraer datos por split ───────────────────────────
    splits = {
        'train': (pkg['train']['params'], pkg['train']['labels']),
        'val':   (pkg['val']['params'],   pkg['val']['labels']),
        'test':  (pkg['test']['params'],  pkg['test']['labels']),
    }

    # Resumen
    for name, (params, labels) in splits.items():
        n_dep = int(np.sum(labels == 1))
        n_nodep = int(np.sum(labels == 0))
        print(f"  {name:>5s}: {len(labels)} muestras "
              f"(dep={n_dep}, no_dep={n_nodep}) "
              f"| params shape={params.shape}")

    total = sum(len(labels) for _, labels in splits.values())
    print(f"\n  Total: {total} imágenes a generar")
    assert total == 1136, f"Se esperan 1136 muestras, hay {total}"

    # ── 3. Generar imágenes por split ────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    t_global = time.time()
    all_metadata = {}

    for split_name, (params_norm, labels) in splits.items():
        metadata = generate_split(
            params_norm=params_norm,
            labels=labels,
            split_name=split_name,
            output_base=OUTPUT_DIR,
        )
        all_metadata[split_name] = metadata

        # ── 4. Guardar CSV de metadata ───────────────────────
        csv_path = os.path.join(
            METADATA_DIR, f"metadata_attractor_{split_name}.csv"
        )
        df = pd.DataFrame(metadata)
        df.to_csv(csv_path, index=False)
        print(f"  → Metadata guardada: {csv_path}")

    # ── 5. Resumen final ─────────────────────────────────────
    elapsed_global = time.time() - t_global

    print(f"\n{'=' * 60}")
    print(f"  RESUMEN FINAL")
    print(f"{'=' * 60}")

    total_generated = 0
    for split_name, metadata in all_metadata.items():
        n = len(metadata)
        total_generated += n
        n_dep = sum(1 for m in metadata if m['label'] == 1)
        n_nodep = n - n_dep
        print(f"  {split_name:>5s}: {n} imágenes (dep={n_dep}, no_dep={n_nodep})")

    print(f"\n  Total generadas: {total_generated} / {total}")
    print(f"  Tiempo total: {elapsed_global:.1f}s "
          f"({elapsed_global/60:.1f} min)")
    print(f"  Directorio: {OUTPUT_DIR}")
    print(f"  Metadata: {METADATA_DIR}/metadata_attractor_*.csv")

    if total_generated == total:
        print(f"\n  ✓ Dataset Fase 1 completo.")
    else:
        print(f"\n  ⚠ Faltan {total - total_generated} imágenes. "
              f"Revisar errores arriba.")


if __name__ == '__main__':
    main()