"""
visualize_attractor.py
──────────────────────
Genera 3 visualizaciones de documentación para el Trabajo Terminal:

  1. Grid comparativo: 3 muestras depresivas (arriba) vs 3 no depresivas (abajo)
  2. Evolución del pipeline: etapas de construcción de una imagen por clase
  3. GIF animado: construcción progresiva de una imagen por clase

Ubicación: src/attractor/visualize_attractor.py

Uso:
  cd proyecto-transformacion-texto-imagen
  python src/attractor/visualize_attractor.py
"""

import os
import sys
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from PIL import Image

# ── Rutas ────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))

RUTA_UMAP_PKL = os.path.join(PROJECT_ROOT, 'data', 'shared', 'umap_params.pkl')
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, 'reports', 'figures')

# ── Importar módulos del pipeline ────────────────────────────
sys.path.insert(0, BASE_DIR)
from lorenz import (
    map_umap_to_lorenz,
    solve_attractor,
    render_attractor_pure,
    CMAP_DEPRESIVO,
    CMAP_NO_DEPRESIVO,
    IMG_SIZE,
)
from render import (
    render_attractor_enriched,
    compute_velocity,
    normalize_array,
    LW_RANGE,
    ALPHA_RANGE,
)

# ── Constantes estéticas ─────────────────────────────────────
BG_COLOR = 'black'


# ═════════════════════════════════════════════════════════════
# UTILIDAD: Renderizar a array numpy (sin guardar a disco)
# ═════════════════════════════════════════════════════════════

def fig_to_array(fig):
    """Convierte una figura matplotlib a un array numpy RGB.
    Usa savefig a buffer en memoria — compatible con todas las versiones."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, facecolor=fig.get_facecolor(),
                bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return np.array(img)


def render_to_array(trajectory, label, elevacion, mode='enriched'):
    """
    Renderiza una trayectoria y retorna como array numpy 256×256×3.

    Parameters
    ----------
    mode : str
        'pure' para Fase 1, 'enriched' para Fase 2.
    """
    if mode == 'pure':
        fig = render_attractor_pure(
            trajectory=trajectory,
            label=label,
            elevacion=elevacion,
        )
    else:
        fig = render_attractor_enriched(
            trajectory=trajectory,
            label=label,
            elevacion=elevacion,
        )
    img = fig_to_array(fig)
    plt.close(fig)

    # Redimensionar a 256×256 por seguridad
    pil_img = Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.array(pil_img)


# ═════════════════════════════════════════════════════════════
# 1. GRID COMPARATIVO DE CLASES
# ═════════════════════════════════════════════════════════════

def generate_class_grid(pkg, output_path, n_samples=3):
    """
    Genera un grid de 2 filas × n_samples columnas:
    Fila superior: muestras depresivas (label=1)
    Fila inferior: muestras no depresivas (label=0)
    """
    print("\n[1/3] Generando grid comparativo de clases...")

    params_train = pkg['train']['params']
    labels_train = pkg['train']['labels']

    # Seleccionar índices por clase
    idx_dep   = np.where(labels_train == 1)[0]
    idx_nodep = np.where(labels_train == 0)[0]

    # Elegir muestras espaciadas (no las primeras, para variedad)
    np.random.seed(42)
    sel_dep   = np.sort(np.random.choice(idx_dep,   n_samples, replace=False))
    sel_nodep = np.sort(np.random.choice(idx_nodep, n_samples, replace=False))

    # Crear figura
    fig, axes = plt.subplots(
        2, n_samples,
        figsize=(4 * n_samples, 8.5),
        facecolor='white',
    )

    # Título principal
    fig.suptitle(
        'Imágenes Generadas por la Técnica de Atractores Extraños (Lorenz)\n'
        'Fila superior: Depresivo (PuBuGn) | Fila inferior: No Depresivo (YlOrRd)',
        fontsize=14, fontweight='bold', y=0.98,
    )

    # Fila superior: depresivo
    for col, idx in enumerate(sel_dep):
        umap_c = params_train[idx]
        label = int(labels_train[idx])
        params = map_umap_to_lorenz(umap_c)

        trajectory = solve_attractor(
            sigma=params['sigma'], rho=params['rho'],
            beta=params['beta'], n_puntos=params['n_puntos'],
        )
        img = render_to_array(trajectory, label, params['elevacion'], mode='enriched')

        ax = axes[0, col]
        ax.imshow(img)
        ax.set_title(
            f"Muestra {idx}\n"
            f"σ={params['sigma']:.1f} | ρ={params['rho']:.1f} | "
            f"β={params['beta']:.1f}",
            fontsize=9, color='black',
        )
        ax.axis('off')

    # Fila inferior: no depresivo
    for col, idx in enumerate(sel_nodep):
        umap_c = params_train[idx]
        label = int(labels_train[idx])
        params = map_umap_to_lorenz(umap_c)

        trajectory = solve_attractor(
            sigma=params['sigma'], rho=params['rho'],
            beta=params['beta'], n_puntos=params['n_puntos'],
        )
        img = render_to_array(trajectory, label, params['elevacion'], mode='enriched')

        ax = axes[1, col]
        ax.imshow(img)
        ax.set_title(
            f"Muestra {idx}\n"
            f"σ={params['sigma']:.1f} | ρ={params['rho']:.1f} | "
            f"β={params['beta']:.1f}",
            fontsize=9, color='black',
        )
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Grid guardado: {output_path}")


# ═════════════════════════════════════════════════════════════
# 2. EVOLUCIÓN DEL PIPELINE (ETAPAS)
# ═════════════════════════════════════════════════════════════

def render_stage_wireframe(trajectory, elevacion):
    """Etapa 1: Trayectoria 3D wireframe (gris, sin color semántico)."""
    fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    ax.plot(
        trajectory['x'], trajectory['y'], trajectory['z'],
        color='white', linewidth=0.3, alpha=0.8,
    )
    ax.view_init(elev=elevacion, azim=45.0)
    ax.set_axis_off()
    ax.grid(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('none')

    img = fig_to_array(fig)
    plt.close(fig)
    return Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def render_stage_color(trajectory, label, elevacion):
    """Etapa 2: Color por z(t) aplicado, grosor uniforme, opacidad 1.0."""
    x, y, z = trajectory['x'], trajectory['y'], trajectory['z']

    z_mid = (z[:-1] + z[1:]) / 2.0
    z_norm = normalize_array(z_mid)

    cmap_name = CMAP_DEPRESIVO if label == 1 else CMAP_NO_DEPRESIVO
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(z_norm)

    points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    lc = Line3DCollection(segments, colors=colors, linewidths=0.3)
    ax.add_collection3d(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.view_init(elev=elevacion, azim=45.0)
    ax.set_axis_off()
    ax.grid(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('none')

    img = fig_to_array(fig)
    plt.close(fig)
    return Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def render_stage_thickness(trajectory, label, elevacion):
    """Etapa 3: Color por z(t) + grosor por velocidad, opacidad 1.0."""
    x, y, z = trajectory['x'], trajectory['y'], trajectory['z']

    speed = compute_velocity(trajectory)
    speed_norm = normalize_array(speed)
    z_mid = (z[:-1] + z[1:]) / 2.0
    z_norm = normalize_array(z_mid)

    cmap_name = CMAP_DEPRESIVO if label == 1 else CMAP_NO_DEPRESIVO
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(z_norm)

    lw_min, lw_max = LW_RANGE
    linewidths = lw_min + speed_norm * (lw_max - lw_min)

    points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    lc = Line3DCollection(segments, colors=colors, linewidths=linewidths)
    ax.add_collection3d(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())
    ax.view_init(elev=elevacion, azim=45.0)
    ax.set_axis_off()
    ax.grid(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('none')

    img = fig_to_array(fig)
    plt.close(fig)
    return Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def render_stage_params(params, umap_c, label):
    """
    Etapa 0: Tarjeta visual de parámetros UMAP → Lorenz.
    Muestra los 5 componentes como barras horizontales con sus valores
    mapeados, sobre fondo negro para mantener consistencia visual.
    """
    from lorenz import PARAM_RANGES

    fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Colormap para las barras según la clase
    cmap_name = CMAP_DEPRESIVO if label == 1 else CMAP_NO_DEPRESIVO
    cmap = plt.get_cmap(cmap_name)

    # Parámetros a mostrar (nombre, valor UMAP, valor mapeado, unidad)
    bar_data = [
        ('σ (sigma)',     umap_c[0], f"{params['sigma']:.1f}",     '[8, 14]'),
        ('ρ (rho)',       umap_c[1], f"{params['rho']:.1f}",       '[24, 32]'),
        ('β (beta)',      umap_c[2], f"{params['beta']:.2f}",      '[2, 4]'),
        ('n_puntos',      umap_c[3], f"{params['n_puntos']:,}",    '[5k, 100k]'),
        ('elevación',     umap_c[4], f"{params['elevacion']:.1f}°", '[0°, 180°]'),
    ]

    n_bars = len(bar_data)
    bar_height = 0.06
    spacing = 0.13
    start_y = 0.82
    bar_left = 0.30
    bar_width = 0.55

    # Título de la tarjeta
    ax.text(0.50, 0.95, 'UMAP → Lorenz', color='white',
            fontsize=9, fontweight='bold', ha='center', va='center',
            family='monospace')

    for i, (name, umap_val, mapped_str, rango) in enumerate(bar_data):
        y = start_y - i * spacing

        # Nombre del parámetro a la izquierda
        ax.text(bar_left - 0.03, y, name, color='white',
                fontsize=6.5, ha='right', va='center', family='monospace')

        # Barra de fondo (gris oscuro)
        ax.barh(y, bar_width, height=bar_height, left=bar_left,
                color='#222222', edgecolor='#444444', linewidth=0.5)

        # Barra de progreso (coloreada según valor UMAP)
        bar_color = cmap(0.3 + umap_val * 0.5)  # rango medio del colormap
        ax.barh(y, bar_width * umap_val, height=bar_height, left=bar_left,
                color=bar_color, edgecolor='none')

        # Valor mapeado a la derecha
        ax.text(bar_left + bar_width + 0.03, y, mapped_str,
                color='white', fontsize=6.5, ha='left', va='center',
                family='monospace', fontweight='bold')

        # Rango debajo de la barra
        ax.text(bar_left + bar_width / 2, y - bar_height * 0.8, rango,
                color='#888888', fontsize=5, ha='center', va='top',
                family='monospace')

    img = fig_to_array(fig)
    plt.close(fig)
    return Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def generate_pipeline_evolution(pkg, output_path):
    """
    Genera imagen de evolución del pipeline con 5 etapas × 2 clases.
    Etapas: Parámetros → Trayectoria → Color z(t) → Grosor v(t) → Opacidad v(t)
    """
    print("\n[2/3] Generando evolución del pipeline...")

    params_train = pkg['train']['params']
    labels_train = pkg['train']['labels']

    stage_names = [
        "Etapa 0\nMapeo UMAP → Lorenz\n(5 componentes)",
        "Etapa 1\nTrayectoria 3D\n(wireframe)",
        "Etapa 2\nColor por z(t)\n(colormap semántico)",
        "Etapa 3\nGrosor por ‖v(t)‖\n(velocidad local)",
        "Etapa 4\nOpacidad por ‖v(t)‖\n(imagen final)",
    ]

    # Seleccionar una muestra de cada clase
    idx_dep   = np.where(labels_train == 1)[0]
    idx_nodep = np.where(labels_train == 0)[0]
    np.random.seed(42)
    sample_dep   = np.random.choice(idx_dep)
    sample_nodep = np.random.choice(idx_nodep)

    samples = [
        (sample_dep,   1, 'Depresivo'),
        (sample_nodep, 0, 'No Depresivo'),
    ]

    fig, axes = plt.subplots(
        2, 5,
        figsize=(20, 8.5),
        facecolor='white',
    )

    fig.suptitle(
        'Pipeline de Generación — Atractores Extraños (Lorenz)\n'
        'Evolución completa: parámetros UMAP → imagen enriquecida final',
        fontsize=14, fontweight='bold', y=0.98,
    )

    for row, (idx, label, clase_name) in enumerate(samples):
        umap_c = params_train[idx]
        params = map_umap_to_lorenz(umap_c)
        elevacion = params['elevacion']

        trajectory = solve_attractor(
            sigma=params['sigma'], rho=params['rho'],
            beta=params['beta'], n_puntos=params['n_puntos'],
        )

        # Etapa 0: Tarjeta de parámetros
        img0 = render_stage_params(params, umap_c, label)

        # Etapa 1: Wireframe
        img1 = render_stage_wireframe(trajectory, elevacion)

        # Etapa 2: Color por z(t)
        img2 = render_stage_color(trajectory, label, elevacion)

        # Etapa 3: Color + grosor
        img3 = render_stage_thickness(trajectory, label, elevacion)

        # Etapa 4: Enriquecida completa (color + grosor + opacidad)
        img4_arr = render_to_array(trajectory, label, elevacion, mode='enriched')
        img4 = Image.fromarray(img4_arr)

        stages = [img0, img1, img2, img3, img4]

        for col, (stage_img, stage_name) in enumerate(zip(stages, stage_names)):
            ax = axes[row, col]
            ax.imshow(np.array(stage_img))
            ax.axis('off')

            if row == 0:
                ax.set_title(stage_name, fontsize=10, fontweight='bold', pad=10)

        # Etiqueta de clase a la izquierda
        axes[row, 0].text(
            -0.15, 0.5,
            f"Muestra {idx}\nClase: {clase_name}",
            transform=axes[row, 0].transAxes,
            fontsize=10, fontweight='bold',
            ha='center', va='center',
            rotation=90, color='black',
        )

    plt.tight_layout(rect=[0.05, 0, 1, 0.92])
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  ✓ Evolución guardada: {output_path}")


# ═════════════════════════════════════════════════════════════
# 3. GIF ANIMADO DE CONSTRUCCIÓN PROGRESIVA
# ═════════════════════════════════════════════════════════════

def render_partial_enriched(trajectory, label, elevacion, fraction):
    """
    Renderiza solo una fracción de la trayectoria enriquecida.

    Parameters
    ----------
    fraction : float
        Fracción de la trayectoria a dibujar (0.0 a 1.0).
    """
    x = trajectory['x']
    y = trajectory['y']
    z = trajectory['z']
    t = trajectory['t']
    n_total = len(x)
    n_draw = max(2, int(n_total * fraction))

    # Recortar trayectoria
    traj_partial = {
        'x': x[:n_draw], 'y': y[:n_draw],
        'z': z[:n_draw], 't': t[:n_draw],
    }

    speed = compute_velocity(traj_partial)
    speed_norm = normalize_array(speed)

    z_mid = (z[:n_draw][:-1] + z[:n_draw][1:]) / 2.0
    z_norm = normalize_array(z_mid)

    cmap_name = CMAP_DEPRESIVO if label == 1 else CMAP_NO_DEPRESIVO
    cmap = plt.get_cmap(cmap_name)
    colors_rgba = cmap(z_norm)

    alpha_min, alpha_max = ALPHA_RANGE
    alphas = alpha_min + speed_norm * (alpha_max - alpha_min)
    colors_rgba[:, 3] = alphas

    lw_min, lw_max = LW_RANGE
    linewidths = lw_min + speed_norm * (lw_max - lw_min)

    points = np.column_stack([
        traj_partial['x'], traj_partial['y'], traj_partial['z']
    ]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Usar los mismos límites del atractor completo para estabilidad visual
    fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    lc = Line3DCollection(segments, colors=colors_rgba, linewidths=linewidths)
    ax.add_collection3d(lc)

    # Límites fijos al atractor completo
    ax.set_xlim(trajectory['x'].min(), trajectory['x'].max())
    ax.set_ylim(trajectory['y'].min(), trajectory['y'].max())
    ax.set_zlim(trajectory['z'].min(), trajectory['z'].max())

    ax.view_init(elev=elevacion, azim=45.0)
    ax.set_axis_off()
    ax.grid(False)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('none')

    img = fig_to_array(fig)
    plt.close(fig)
    return Image.fromarray(img).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def generate_construction_gif(pkg, output_path_dep, output_path_nodep,
                              n_frames=40, duration_ms=100):
    """
    Genera GIFs animados mostrando la construcción progresiva del atractor.
    Un GIF por clase.
    """
    print("\n[3/3] Generando GIFs animados de construcción...")

    params_train = pkg['train']['params']
    labels_train = pkg['train']['labels']

    idx_dep   = np.where(labels_train == 1)[0]
    idx_nodep = np.where(labels_train == 0)[0]
    np.random.seed(42)
    sample_dep   = np.random.choice(idx_dep)
    sample_nodep = np.random.choice(idx_nodep)

    samples = [
        (sample_dep,   1, 'depresivo',    output_path_dep),
        (sample_nodep, 0, 'no_depresivo', output_path_nodep),
    ]

    # Fracciones de la trayectoria a dibujar (crecimiento no lineal)
    fractions = np.linspace(0.02, 1.0, n_frames) ** 0.7

    for idx, label, clase_name, out_path in samples:
        print(f"  Generando GIF [{clase_name}] (muestra {idx})...")

        umap_c = params_train[idx]
        params = map_umap_to_lorenz(umap_c)
        elevacion = params['elevacion']

        trajectory = solve_attractor(
            sigma=params['sigma'], rho=params['rho'],
            beta=params['beta'], n_puntos=params['n_puntos'],
        )

        frames = []
        for i, frac in enumerate(fractions):
            frame = render_partial_enriched(
                trajectory, label, elevacion, frac,
            )
            frames.append(frame)

            if (i + 1) % 10 == 0:
                print(f"    Frame {i+1}/{n_frames} ({frac:.0%})")

        # Agregar frames extra al final (pausa en imagen completa)
        for _ in range(8):
            frames.append(frames[-1])

        # Guardar GIF
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
        print(f"  ✓ GIF guardado: {out_path}")


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Visualizaciones de Documentación — Atractores de Lorenz")
    print("=" * 60)

    # Cargar PKL
    assert os.path.exists(RUTA_UMAP_PKL), \
        f"No se encontró umap_params.pkl en: {RUTA_UMAP_PKL}"
    pkg = joblib.load(RUTA_UMAP_PKL)
    print(f"✓ umap_params.pkl cargado")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Grid comparativo
    generate_class_grid(
        pkg,
        output_path=os.path.join(OUTPUT_DIR, 'attractor_class_grid.png'),
        n_samples=3,
    )

    # 2. Evolución del pipeline
    generate_pipeline_evolution(
        pkg,
        output_path=os.path.join(OUTPUT_DIR, 'attractor_pipeline_evolution.png'),
    )

    # 3. GIFs animados
    generate_construction_gif(
        pkg,
        output_path_dep=os.path.join(OUTPUT_DIR, 'attractor_construction_depresivo.gif'),
        output_path_nodep=os.path.join(OUTPUT_DIR, 'attractor_construction_no_depresivo.gif'),
        n_frames=40,
        duration_ms=100,
    )

    print(f"\n{'=' * 60}")
    print(f"  ✓ Todas las visualizaciones generadas en: {OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()