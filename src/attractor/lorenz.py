"""
lorenz.py
─────────
Fase 1 del pipeline de Atractores Extraños.
Genera imágenes "puras" (baseline) a partir de los componentes UMAP:
  • Trazo uniforme, color fijo por clase, grosor constante, opacidad 1.0.
  • Sirve como control experimental antes del enriquecimiento (Fase 2).

Ubicación: src/attractor/lorenz.py

Contrato técnico:
  - 256×256 px, PNG, RGB
  - Paleta depresivo (label=1): PuBuGn
  - Paleta no_depresivo (label=0): YlOrRd
  - Fondo negro para maximizar contraste del atractor.

Mapeo UMAP → Lorenz:
  umap[0] → σ (sigma)   ∈ [8, 14]
  umap[1] → ρ (rho)     ∈ [24, 32]
  umap[2] → β (beta)    ∈ [2, 4]
  umap[3] → n_puntos    ∈ [5_000, 100_000]
  umap[4] → elevación   ∈ [0°, 180°]
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')  # backend sin GUI — compatible con Colab y headless
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

# ─────────────────────────────────────────────────────────────
# 1. Sistema de ecuaciones diferenciales de Lorenz
# ─────────────────────────────────────────────────────────────

def lorenz_system(t, state, sigma, rho, beta):
    """
    Sistema de Lorenz en forma estándar.
    
    dx/dt = σ(y − x)
    dy/dt = x(ρ − z) − y
    dz/dt = xy − βz
    
    Parameters
    ----------
    t : float
        Variable temporal (requerida por solve_ivp, no usada explícitamente).
    state : array-like, shape (3,)
        Estado actual [x, y, z].
    sigma, rho, beta : float
        Parámetros del sistema de Lorenz.
    
    Returns
    -------
    list of float
        Derivadas [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


# ─────────────────────────────────────────────────────────────
# 2. Mapeo de componentes UMAP a parámetros de Lorenz
# ─────────────────────────────────────────────────────────────

# Rangos del contrato técnico
PARAM_RANGES = {
    'sigma':     (8.0,    14.0),
    'rho':       (24.0,   32.0),
    'beta':      (2.0,    4.0),
    'n_puntos':  (5_000,  100_000),
    'elevacion': (0.0,    180.0),
}


def map_umap_to_lorenz(umap_components):
    """
    Convierte 5 valores UMAP normalizados [0, 1] a parámetros físicos
    del atractor de Lorenz.
    
    Parameters
    ----------
    umap_components : array-like, shape (5,)
        Valores normalizados c₀..c₄ del PKL compartido.
    
    Returns
    -------
    dict
        Parámetros mapeados: sigma, rho, beta, n_puntos, elevacion.
    """
    c = np.asarray(umap_components, dtype=np.float64)
    assert c.shape == (5,), f"Se esperan 5 componentes, recibidos {c.shape}"
    assert np.all((c >= 0.0) & (c <= 1.0)), \
        f"Componentes fuera de [0,1]: min={c.min():.6f}, max={c.max():.6f}"

    def _lerp(val, low, high):
        """Interpolación lineal: val ∈ [0,1] → [low, high]."""
        return low + val * (high - low)

    return {
        'sigma':     _lerp(c[0], *PARAM_RANGES['sigma']),
        'rho':       _lerp(c[1], *PARAM_RANGES['rho']),
        'beta':      _lerp(c[2], *PARAM_RANGES['beta']),
        'n_puntos':  int(_lerp(c[3], *PARAM_RANGES['n_puntos'])),
        'elevacion': _lerp(c[4], *PARAM_RANGES['elevacion']),
    }


# ─────────────────────────────────────────────────────────────
# 3. Integración numérica (solver ODE)
# ─────────────────────────────────────────────────────────────

def solve_attractor(sigma, rho, beta, n_puntos,
                    t_max=50.0, initial_state=None, t_discard=5.0):
    """
    Integra el sistema de Lorenz y retorna la trayectoria estable.
    
    Parameters
    ----------
    sigma, rho, beta : float
        Parámetros del sistema de Lorenz.
    n_puntos : int
        Cantidad total de puntos a evaluar (controla densidad del trazo).
    t_max : float, default 50.0
        Tiempo máximo de integración.
    initial_state : array-like or None
        Condición inicial [x₀, y₀, z₀]. Si None, usa [1.0, 1.0, 1.0].
    t_discard : float, default 5.0
        Tiempo inicial a descartar (transitorio antes de que el atractor
        converja a su forma estable).
    
    Returns
    -------
    dict
        'x', 'y', 'z': arrays de la trayectoria (sin transitorio).
        't': array de tiempos correspondientes.
        'params': dict con sigma, rho, beta usados.
    """
    if initial_state is None:
        initial_state = [1.0, 1.0, 1.0]

    # Puntos de evaluación incluyendo transitorio
    n_total = int(n_puntos * (t_max / (t_max - t_discard)))
    t_eval = np.linspace(0, t_max, n_total)

    sol = solve_ivp(
        fun=lorenz_system,
        t_span=(0, t_max),
        y0=initial_state,
        args=(sigma, rho, beta),
        method='RK45',
        t_eval=t_eval,
        rtol=1e-9,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"solve_ivp falló: {sol.message}")

    # Descartar transitorio
    mask = sol.t >= t_discard
    return {
        'x': sol.y[0, mask],
        'y': sol.y[1, mask],
        'z': sol.y[2, mask],
        't': sol.t[mask],
        'params': {'sigma': sigma, 'rho': rho, 'beta': beta},
    }


# ─────────────────────────────────────────────────────────────
# 4. Renderizado Fase 1 — imagen "pura" (baseline)
# ─────────────────────────────────────────────────────────────

# Colormaps del contrato técnico
#CMAP_DEPRESIVO     = 'PuBuGn'   # label = 1
#CMAP_NO_DEPRESIVO  = 'YlOrRd'   # label = 0

# Resolución del contrato técnico
IMG_SIZE = 256


def render_attractor_pure(trajectory, elevacion,
                          output_path=None, azimuth=45.0,
                          linewidth=0.3, bg_color='black'):
    """
    Renderiza la trayectoria como imagen 256×256 PNG RGB — Fase 1 (pura).
    
    Características del renderizado puro (baseline):
      - Color FIJO: tono medio del colormap de la clase.
      - Grosor CONSTANTE: linewidth uniforme en todo el trazo.
      - Opacidad 1.0: sin transparencia.
      - Proyección 3D con ángulo de elevación controlado por UMAP.
    
    Parameters
    ----------
    trajectory : dict
        Salida de solve_attractor() con claves 'x', 'y', 'z'.
    label : int
        0 = no_depresivo (YlOrRd), 1 = depresivo (PuBuGn).
    elevacion : float
        Ángulo de elevación en grados [0, 180], mapeado desde umap[4].
    output_path : str or None
        Ruta completa del PNG de salida. Si None, no guarda (útil para debug).
    azimuth : float, default 45.0
        Ángulo azimutal fijo (no controlado por UMAP).
    linewidth : float, default 0.3
        Grosor uniforme del trazo.
    bg_color : str, default 'black'
        Color de fondo de la imagen.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura generada (para inspección o cierre manual).
    """
    # Seleccionar colormap según etiqueta
    #cmap_name = CMAP_DEPRESIVO if label == 1 else CMAP_NO_DEPRESIVO
    #cmap = plt.get_cmap(cmap_name)

    # Color fijo: tono medio del colormap (valor 0.5)
    #color_fijo = cmap(0.5)

    # Crear figura cuadrada sin bordes
    fig = plt.figure(figsize=(IMG_SIZE / 100, IMG_SIZE / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1], projection='3d')

    # Fondo
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Dibujar trayectoria con trazo uniforme
    ax.plot(
        trajectory['x'],
        trajectory['y'],
        trajectory['z'],
    #   color=color_fijo,
        linewidth=linewidth,
        alpha=1.0,
    )

    # Configurar vista
    ax.view_init(elev=elevacion, azim=azimuth)

    # Eliminar ejes, grid, bordes — solo el atractor
    ax.set_axis_off()
    ax.grid(False)

    # Eliminar paneles de fondo del 3D
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')

# Guardar si se proporcionó ruta
    if output_path is not None:
        directorio = os.path.dirname(output_path)
        # Solo intentar crear el directorio si la ruta contiene uno
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
# 5. Pipeline completo Fase 1: UMAP → Imagen pura
# ─────────────────────────────────────────────────────────────

def generate_pure_image(umap_components, output_path=None):
    """
    Pipeline completo de Fase 1 para UNA muestra.
    
    Parameters
    ----------
    umap_components : array-like, shape (5,)
        Componentes normalizados [0, 1] del PKL compartido.
    label : int
        0 = no_depresivo, 1 = depresivo.
    output_path : str or None
        Ruta del PNG de salida.
    
    Returns
    -------
    dict
        'params': parámetros de Lorenz mapeados.
        'trajectory': trayectoria calculada.
        'fig': figura de matplotlib.
    """
    # Paso 1: Mapear UMAP → parámetros
    params = map_umap_to_lorenz(umap_components)

    # Paso 2: Resolver ODE
    trajectory = solve_attractor(
        sigma=params['sigma'],
        rho=params['rho'],
        beta=params['beta'],
        n_puntos=params['n_puntos'],
    )

    # Paso 3: Renderizar imagen pura
    fig = render_attractor_pure(
        trajectory=trajectory,
        elevacion=params['elevacion'],
        output_path=output_path,
    )

    return {
        'params': params,
        'trajectory': trajectory,
        'fig': fig,
    }


# ─────────────────────────────────────────────────────────────
# 6. Quick test — ejecutar directamente para verificar
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("Test rápido — Lorenz Fase 1 (imagen pura)")
    print("=" * 60)

    # Simular un vector UMAP con parámetros clásicos de Lorenz:
    # σ=10 → (10-8)/6 = 0.333
    # ρ=28 → (28-24)/8 = 0.500
    # β=8/3 ≈ 2.667 → (2.667-2)/2 = 0.333
    # n_puntos=50000 → (50000-5000)/95000 = 0.474
    # elevación=30° → 30/180 = 0.167
    test_umap = np.array([0.333, 0.500, 0.333, 0.474, 0.167])

    # Test para ambas clases
    for lbl, nombre in [(1, 'depresivo'), (0, 'no_depresivo')]:
        result = generate_pure_image(
            umap_components=test_umap,
            output_path=f'test_lorenz_{nombre}.png',
        )
        p = result['params']
        traj = result['trajectory']
        print(f"\n  [{nombre}]")
        print(f"  σ={p['sigma']:.3f}, ρ={p['rho']:.3f}, β={p['beta']:.3f}")
        print(f"  n_puntos={p['n_puntos']}, elevación={p['elevacion']:.1f}°")
        print(f"  Trayectoria: {len(traj['x'])} puntos")
        print(f"  Rango x: [{traj['x'].min():.2f}, {traj['x'].max():.2f}]")
        print(f"  Rango z: [{traj['z'].min():.2f}, {traj['z'].max():.2f}]")
        print(f"  → Imagen guardada: test_lorenz_{nombre}.png")
        plt.close(result['fig'])  # liberar memoria

    print("\n✓ Test completado — verificar visualmente las imágenes.")