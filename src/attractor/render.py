"""
render.py
─────────
Fase 2 del pipeline de Atractores Extraños — Enriquecimiento visual.
Aplica capas estéticas derivadas de la trayectoria ya calculada,
sin consumir componentes UMAP adicionales.

Ubicación: src/attractor/render.py

Capas estéticas:
  1. Color por segmento:   z(t) normalizado → colormap completo de la clase
  2. Grosor por segmento:  ‖v(t)‖ normalizado → linewidth ∈ [0.1, 1.5]
  3. Opacidad por segmento: ‖v(t)‖ normalizado → alpha ∈ [0.3, 0.7]

Contrato técnico:
  - 256×256 px, PNG, RGB
  - Paleta depresivo (label=1): PuBuGn
  - Paleta no_depresivo (label=0): YlOrRd
  - Fondo negro
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import os


# Colormaps del contrato técnico
CMAP_DEPRESIVO    = 'PuBuGn'   # label = 1
CMAP_NO_DEPRESIVO = 'YlOrRd'   # label = 0

# Resolución del contrato técnico
IMG_SIZE = 256

# Rangos de las capas estéticas
LW_RANGE    = (0.1, 1.5)    # grosor mínimo y máximo
ALPHA_RANGE = (0.3, 0.7)    # opacidad mínima y máxima


def compute_velocity(trajectory):
    """
    Calcula la velocidad local ‖v(t)‖ en cada punto de la trayectoria.

    La velocidad se calcula como la norma euclidiana del vector de
    diferencias finitas entre puntos consecutivos.

    Parameters
    ----------
    trajectory : dict
        Salida de solve_attractor() con claves 'x', 'y', 'z', 't'.

    Returns
    -------
    ndarray, shape (N-1,)
        Velocidad local en cada segmento (entre punto i y punto i+1).
    """
    dx = np.diff(trajectory['x'])
    dy = np.diff(trajectory['y'])
    dz = np.diff(trajectory['z'])
    dt = np.diff(trajectory['t'])

    # Evitar división por cero en caso de dt=0
    dt = np.where(dt == 0, 1e-12, dt)

    speed = np.sqrt(dx**2 + dy**2 + dz**2) / dt
    return speed


def normalize_array(arr):
    """
    Normaliza un array al rango [0, 1].

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    ndarray
        Valores normalizados. Si min == max, retorna array de 0.5.
    """
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - vmin) / (vmax - vmin)


def render_attractor_enriched(trajectory, elevacion,
                              output_path=None, azimuth=45.0,
                              lw_range=LW_RANGE,
                              alpha_range=ALPHA_RANGE,
                              invert_alpha=False,
                              bg_color='black'):
    """
    Renderiza la trayectoria como imagen 256×256 PNG RGB — Fase 2 (enriquecida).

    Cada segmento de la trayectoria tiene su propio color, grosor y opacidad,
    derivados directamente de la geometría y dinámica del atractor.

    Parameters
    ----------
    trajectory : dict
        Salida de solve_attractor() con claves 'x', 'y', 'z', 't'.
    label : int
        0 = no_depresivo (YlOrRd), 1 = depresivo (PuBuGn).
    elevacion : float
        Ángulo de elevación en grados [0, 180].
    output_path : str or None
        Ruta completa del PNG de salida.
    azimuth : float, default 45.0
        Ángulo azimutal fijo.
    lw_range : tuple, default (0.1, 1.5)
        Rango de grosor (min, max).
    alpha_range : tuple, default (0.3, 0.7)
        Rango de opacidad (min, max).
    invert_alpha : bool, default False
        Si True, velocidad alta → opacidad BAJA (resalta zonas lentas).
        Si False, velocidad alta → opacidad ALTA (resalta zonas rápidas).
    bg_color : str, default 'black'
        Color de fondo.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura generada.
    """
    # ── 1. Calcular propiedades por segmento ─────────────────
    speed = compute_velocity(trajectory)
    speed_norm = normalize_array(speed)

    # z normalizado para color (valor medio de cada segmento)
    z = trajectory['z']
    z_mid = (z[:-1] + z[1:]) / 2.0
    z_norm = normalize_array(z_mid)

    # ── 2. Mapear a capas estéticas ──────────────────────────
    # Color: z_norm → colormap completo
    #cmap_name = CMAP_DEPRESIVO if label == 1 else CMAP_NO_DEPRESIVO
    #cmap = plt.get_cmap(cmap_name)
    #colors_rgba = cmap(z_norm)  # shape (N-1, 4)

    # Opacidad: velocidad → alpha
    alpha_min, alpha_max = alpha_range
    if invert_alpha:
        alphas = alpha_max - speed_norm * (alpha_max - alpha_min)
    else:
        alphas = alpha_min + speed_norm * (alpha_max - alpha_min)
    #colors_rgba[:, 3] = alphas

    # Grosor: velocidad → linewidth
    lw_min, lw_max = lw_range
    linewidths = lw_min + speed_norm * (lw_max - lw_min)

    # ── 3. Construir segmentos 3D ────────────────────────────
    x = trajectory['x']
    y = trajectory['y']

    # Cada segmento es un par de puntos [[x_i,y_i,z_i], [x_{i+1},y_{i+1},z_{i+1}]]
    points = np.column_stack([x, y, z]).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)  # (N-1, 2, 3)

    # ── 4. Renderizar con Line3DCollection ───────────────────
    fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    # Fondo
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Crear colección de líneas 3D
    lc = Line3DCollection(
        segments,
        #colors=colors_rgba,
        linewidths=linewidths,
    )
    ax.add_collection3d(lc)

    # Ajustar límites de los ejes al rango de datos
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_zlim(z.min(), z.max())

    # Configurar vista
    ax.view_init(elev=elevacion, azim=azimuth)

    # Eliminar ejes, grid, bordes
    ax.set_axis_off()
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

    # Guardar
    if output_path is not None:
        directorio = os.path.dirname(output_path)
        if directorio:
            os.makedirs(directorio, exist_ok=True)

        fig.savefig(
            output_path,
            dpi=100,
            facecolor=bg_color,
            bbox_inches='tight',
            pad_inches=0,
        )

    return fig


# ─────────────────────────────────────────────────────────────
# Pipeline completo Fase 2: UMAP → Imagen enriquecida
# ─────────────────────────────────────────────────────────────

def generate_enriched_image(umap_components, output_path=None,
                            invert_alpha=False):
    """
    Pipeline completo de Fase 2 para UNA muestra.

    Reutiliza map_umap_to_lorenz y solve_attractor de lorenz.py,
    luego aplica el renderizado enriquecido.

    Parameters
    ----------
    umap_components : array-like, shape (5,)
        Componentes normalizados [0, 1].
    label : int
        0 = no_depresivo, 1 = depresivo.
    output_path : str or None
        Ruta del PNG de salida.
    invert_alpha : bool, default False
        Dirección del mapeo de opacidad.

    Returns
    -------
    dict
        'params', 'trajectory', 'fig'.
    """
    from lorenz import map_umap_to_lorenz, solve_attractor

    # Paso 1: Mapear UMAP → parámetros
    params = map_umap_to_lorenz(umap_components)

    # Paso 2: Resolver ODE (misma trayectoria que Fase 1)
    trajectory = solve_attractor(
        sigma=params['sigma'],
        rho=params['rho'],
        beta=params['beta'],
        n_puntos=params['n_puntos'],
    )

    # Paso 3: Renderizar con enriquecimiento
    fig = render_attractor_enriched(
        trajectory=trajectory,
        elevacion=params['elevacion'],
        output_path=output_path,
        invert_alpha=invert_alpha,
    )

    return {
        'params': params,
        'trajectory': trajectory,
        'fig': fig,
    }


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import numpy as np

    print("=" * 60)
    print("Test rápido — Lorenz Fase 2 (imagen enriquecida)")
    print("=" * 60)

    # Parámetros clásicos de Lorenz (mismos que test de Fase 1)
    test_umap = np.array([0.333, 0.500, 0.333, 0.474, 0.167])

    for lbl, nombre in [(1, 'depresivo'), (0, 'no_depresivo')]:
        result = generate_enriched_image(
            umap_components=test_umap,
            label=lbl,
            output_path=f'test_lorenz_enriched_{nombre}.png',
        )
        p = result['params']
        traj = result['trajectory']
        speed = compute_velocity(traj)
        print(f"\n  [{nombre}]")
        print(f"  σ={p['sigma']:.3f}, ρ={p['rho']:.3f}, β={p['beta']:.3f}")
        print(f"  Velocidad: min={speed.min():.2f}, "
              f"max={speed.max():.2f}, mean={speed.mean():.2f}")
        print(f"  → Imagen guardada: test_lorenz_enriched_{nombre}.png")
        plt.close(result['fig'])

    print("\n✓ Test completado — comparar visualmente con Fase 1.")